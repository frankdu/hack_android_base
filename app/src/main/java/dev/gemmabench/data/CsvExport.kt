package dev.gemmabench.data

import dev.gemmabench.benchmark.BenchmarkResult

object CsvExport {
    fun encode(results: List<BenchmarkResult>): String {
        val rows = mutableListOf(listOf(
            "result_id", "started_at", "model", "sha256", "model_bytes", "device", "soc", "android",
            "fingerprint", "runtime", "backend", "workload", "mtp", "requested_input", "requested_output",
            "context_setting", "status", "error", "run", "warmup", "native_input_tokens", "native_output_tokens",
            "native_init_s", "native_ttft_s", "prefill_tokens_s", "decode_tokens_s", "elapsed_ms",
            "engine_load_ms", "conversation_setup_ms", "first_visible_text_ms", "engine_reused",
            "thermal_before", "thermal_after", "battery_temp_before_c", "battery_temp_after_c",
            "pss_before_mb", "pss_after_mb", "prompt", "output",
        ))
        results.forEach { r ->
            val prefix = listOf(
                r.id, r.startedAt, r.model.slot.name, r.model.sha256, r.model.bytes,
                r.device.model, r.device.soc, r.device.androidRelease, r.device.fingerprint, r.device.runtimeVersion,
                r.settings.accelerator.name, r.settings.workload.name, r.settings.speculativeDecoding,
                r.settings.prefillTokens, r.settings.outputTokens, r.settings.contextTokens, r.status, r.error,
            )
            if (r.samples.isEmpty()) {
                rows += (prefix + List(19) { null } + r.settings.prompt + null).map { it?.toString().orEmpty() }
            }
            r.samples.forEach { s ->
                rows += (prefix + listOf(
                    s.index, s.warmup, s.metrics.prefillTokens, s.metrics.decodeTokens,
                    s.metrics.initSeconds, s.metrics.ttftSeconds,
                    s.metrics.prefillTokensPerSecond, s.metrics.decodeTokensPerSecond,
                    s.elapsedMs, s.engineLoadMs, s.conversationSetupMs, s.firstVisibleTextMs, s.engineReused,
                    s.before.thermalStatus, s.after.thermalStatus,
                    s.before.batteryTemperatureCelsius, s.after.batteryTemperatureCelsius,
                    s.before.processPssMb, s.after.processPssMb, r.settings.prompt, s.output,
                )).map { it?.toString().orEmpty() }
            }
        }
        return rows.joinToString("\r\n", postfix = "\r\n") { row -> row.joinToString(",") { escape(it) } }
    }

    // Quote every field, preserve multiline text, and avoid spreadsheet formula execution.
    fun escape(value: String): String {
        val safe = if (value.trimStart().firstOrNull() in listOf('=', '+', '-', '@', '\t', '\r')) "'$value" else value
        return "\"${safe.replace("\"", "\"\"")}\""
    }
}
