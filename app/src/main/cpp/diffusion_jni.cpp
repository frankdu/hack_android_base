#include <jni.h>
#include <android/log.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "stable-diffusion.h"
#ifdef WORD_DIFFUSION_VULKAN
#include "ggml-vulkan.h"
#endif

namespace {
std::mutex context_mutex;
sd_ctx_t* active_context = nullptr;
std::atomic<bool> stopped{false};
struct UtfString {
    JNIEnv* env; jstring value; const char* chars;
    UtfString(JNIEnv* e, jstring s) : env(e), value(s), chars(s ? e->GetStringUTFChars(s, nullptr) : "") {}
    ~UtfString() { if (value) env->ReleaseStringUTFChars(value, chars); }
};
void log_callback(sd_log_level_t level, const char* message, void*) {
    // Model logs can include tokenized prompts: keep logcat to warnings/errors only.
    if (level >= SD_LOG_WARN) __android_log_write(ANDROID_LOG_WARN, "WordDiffusion", message);
}
struct Progress { JNIEnv* env; jobject listener; jmethodID method; };
void progress_callback(int step, int steps, float seconds, void* data) {
    auto* p = static_cast<Progress*>(data);
    p->env->CallVoidMethod(p->listener, p->method, step, steps, seconds);
}
}

extern "C" JNIEXPORT void JNICALL
Java_dev_gemmabench_imagegen_DiffusionNative_resetStop(JNIEnv*, jobject) { stopped.store(false); }

extern "C" JNIEXPORT void JNICALL
Java_dev_gemmabench_imagegen_DiffusionNative_requestStop(JNIEnv*, jobject) {
    stopped.store(true);
    std::lock_guard<std::mutex> lock(context_mutex);
    if (active_context) sd_cancel_generation(active_context, SD_CANCEL_ALL);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_dev_gemmabench_imagegen_DiffusionNative_generateOnBackend(JNIEnv* env, jobject,
        jstring model_path, jstring encoder_path, jstring vae_path, jstring prompt,
        jboolean flux, jboolean gpu, jint steps, jlong seed, jint threads, jobject listener) {
    const auto features = getauxval(AT_HWCAP);
    if (!(features & HWCAP_ASIMDHP) || !(features & HWCAP_ASIMDDP)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Image generation requires ARM FP16 and dot-product instructions (supported by Pixel 10/11).");
        return nullptr;
    }
    UtfString model(env, model_path), encoder(env, encoder_path), vae(env, vae_path), text(env, prompt);
    if (gpu) {
#ifdef WORD_DIFFUSION_VULKAN
        if (ggml_backend_vk_get_device_count() < 1) {
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "No Vulkan GPU is available. GPU was not rerouted to CPU.");
            return nullptr;
        }
#else
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "This build does not include Vulkan GPU support.");
        return nullptr;
#endif
    }
    sd_set_log_callback(log_callback, nullptr);
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.n_threads = threads;
    ctx_params.enable_mmap = true;
    // The Pixel PowerVR decoder can stall; compare a reproducible hybrid GPU pipeline.
    ctx_params.backend = gpu ? "te=cpu,diffusion=vulkan0,vae=cpu" : "cpu";
    ctx_params.params_backend = gpu ? nullptr : "cpu";
    ctx_params.flash_attn = true;
    ctx_params.rng_type = CPU_RNG;
    if (flux) {
        ctx_params.diffusion_model_path = model.chars;
        ctx_params.llm_path = encoder.chars;
        ctx_params.vae_path = vae.chars;
    } else {
        ctx_params.model_path = model.chars;
    }
    if (stopped.load()) return nullptr;
    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Could not load the diffusion model. Check model files and available RAM.");
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(context_mutex);
        active_context = ctx;
        if (stopped.load()) sd_cancel_generation(ctx, SD_CANCEL_ALL);
    }
    Progress progress{env, listener, env->GetMethodID(env->GetObjectClass(listener), "onProgress", "(IIF)V")};
    sd_set_progress_callback(progress_callback, &progress);
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = text.chars;
    params.negative_prompt = "";
    params.width = 512; params.height = 512;
    params.seed = seed; params.batch_count = 1;
    params.sample_params.sample_steps = steps;
    params.sample_params.guidance.txt_cfg = 1.0f; // sd.cpp's 1 disables CFG; Diffusers expresses this as 0.
    params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    params.sample_params.scheduler = flux ? FLUX2_SCHEDULER : SGM_UNIFORM_SCHEDULER;
    params.vae_tiling_params.enabled = true;
    params.vae_tiling_params.tile_size_w = 256;
    params.vae_tiling_params.tile_size_h = 256;
    sd_image_t* images = nullptr;
    int count = 0;
    bool ok = !stopped.load() && generate_image(ctx, &params, &images, &count);
    jintArray result = nullptr;
    if (ok && count > 0 && images[0].data && !stopped.load() && !env->ExceptionCheck()) {
        const auto& img = images[0];
        std::vector<jint> pixels(img.width * img.height);
        for (size_t i = 0; i < pixels.size(); ++i) {
            const auto* rgb = img.data + i * img.channel;
            pixels[i] = static_cast<jint>(0xff000000u | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]);
        }
        result = env->NewIntArray(static_cast<jsize>(pixels.size()));
        if (result) env->SetIntArrayRegion(result, 0, static_cast<jsize>(pixels.size()), pixels.data());
    }
    if (images) free_sd_images(images, count);
    sd_set_progress_callback(nullptr, nullptr);
    {
        std::lock_guard<std::mutex> lock(context_mutex);
        active_context = nullptr;
    }
    free_sd_ctx(ctx);
    if (!ok && !stopped.load() && !env->ExceptionCheck()) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Image generation failed. The model may exceed available device memory; see WordDiffusion in logcat.");
    }
    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_gemmabench_imagegen_DiffusionNative_vulkanBuilt(JNIEnv*, jobject) {
#ifdef WORD_DIFFUSION_VULKAN
    return true;
#else
    return false;
#endif
}
