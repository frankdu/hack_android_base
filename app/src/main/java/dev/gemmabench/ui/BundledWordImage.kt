package dev.gemmabench.ui

import androidx.compose.foundation.Image
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.material3.MaterialTheme
import dev.gemmabench.data.BundledWordImages
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
internal fun BundledWordImage(word: String, category: String, size: Dp, modifier: Modifier = Modifier) {
    val assets = LocalContext.current.assets
    val maxPixels = with(LocalDensity.current) { size.roundToPx() }
    val bitmap by produceState<ImageBitmap?>(null, assets, word, category, maxPixels) {
        value = null
        value = withContext(Dispatchers.IO) {
            BundledWordImages(assets).bitmap(word, category, maxPixels)?.asImageBitmap()
        }
    }
    bitmap?.let {
        Image(it, contentDescription = "Illustration of $word", contentScale = ContentScale.Fit,
            modifier = modifier.clip(MaterialTheme.shapes.medium))
    }
}
