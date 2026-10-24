plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
    id("androidx.room")
}

room { schemaDirectory("$projectDir/schemas") }

android {
    namespace = "dev.gemmabench"
    compileSdk = 37
    ndkVersion = "27.0.11902837"
    defaultConfig {
        applicationId = "dev.gemmabench"
        minSdk = 31
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { cppFlags += "-O2" } }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "LITERT_LM_VERSION", "\"0.18.0\"")
    }
    buildFeatures { compose = true; buildConfig = true }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    // Package the source dictionary directly so it remains the single source of truth.
    sourceSets["main"].assets.directories += rootProject.file("seed").path
    sourceSets["test"].resources.directories += rootProject.file("seed").path
    sourceSets["test"].resources.directories += file("src/main/assets/tts").path
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { jniLibs.useLegacyPackaging = true }
    // Optional, version-matched Tensor dispatch libraries. See docs/BENCHMARKING.md.
    providers.gradleProperty("tensorNpuLibDir").orNull?.let {
        sourceSets["main"].jniLibs.directories += it
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2026.08.00"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.18.0")
    // Includes the Tensor dispatcher and optional vendor-library manifest declarations.
    // A custom, version-matched runtime directory can still override the packaged runtime.
    if (!providers.gradleProperty("tensorNpuLibDir").isPresent) {
        runtimeOnly("com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.0")
    }
    implementation("com.google.code.gson:gson:2.13.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")
    implementation("org.opencv:opencv:4.13.0")
    implementation("androidx.room:room-runtime:2.8.5")
    ksp("androidx.room:room-compiler:2.8.5")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation(platform("androidx.compose:compose-bom:2026.08.00"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
}
