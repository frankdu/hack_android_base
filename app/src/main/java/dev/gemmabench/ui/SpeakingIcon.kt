package dev.gemmabench.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.dp
import dev.gemmabench.R

/** Outward-moving, fading waves run only while audio is actually playing. */
@Composable
internal fun SpeakingIcon(playing: Boolean, contentDescription: String) {
    if (!playing) {
        Icon(painterResource(R.drawable.ic_speak_24), contentDescription = contentDescription)
        return
    }
    val transition = rememberInfiniteTransition(label = "Speech waves")
    val phase by transition.animateFloat(0f, 1f,
        infiniteRepeatable(tween(1000, easing = LinearEasing)), label = "Outward wave")
    val color = LocalContentColor.current
    Canvas(Modifier.size(28.dp).semantics {
        this.contentDescription = contentDescription
        stateDescription = "Playing"
    }) {
        scale(size.width / 24f, size.height / 24f, pivot = Offset.Zero) {
            drawPath(Path().apply {
                moveTo(3f, 9f); lineTo(7f, 9f); lineTo(12f, 5f)
                lineTo(12f, 19f); lineTo(7f, 15f); lineTo(3f, 15f); close()
            }, color)
            repeat(3) { index ->
                val progress = (phase + index / 3f) % 1f
                val radius = 3f + progress * 8f
                drawArc(color.copy(alpha = (1f - progress) * 0.95f), -55f, 110f, false,
                    topLeft = Offset(11f - radius, 12f - radius), size = Size(radius * 2f, radius * 2f),
                    style = Stroke(width = 1.8f, cap = StrokeCap.Round))
            }
        }
    }
}
