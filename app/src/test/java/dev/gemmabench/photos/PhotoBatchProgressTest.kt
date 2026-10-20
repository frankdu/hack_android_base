package dev.gemmabench.photos

import org.junit.Assert.*
import org.junit.Test

class PhotoBatchProgressTest {
    private fun state(vararg statuses: PhotoStatus) = PhotoDemoState(results = statuses.mapIndexed { index, status ->
        PhotoResult(DevicePhoto(index.toLong(), "content://photos/$index", "photo-$index.jpg"), status = status)
    })

    @Test fun progressCountsTerminalResultsButNotActiveOrPendingPhotos() {
        val progress = state(PhotoStatus.COMPLETED, PhotoStatus.FAILED, PhotoStatus.ANALYZING, PhotoStatus.PENDING).batchProgress
        assertEquals(0.5f, progress.fraction, 0f)
        assertEquals("2 of 4 photos processed · 1 failed", progress.label)
    }

    @Test fun stoppedAndResumedPhotoDoesNotMoveProgressUntilItFinishes() {
        val stopped = state(PhotoStatus.COMPLETED, PhotoStatus.PENDING).batchProgress
        val resumed = state(PhotoStatus.COMPLETED, PhotoStatus.ANALYZING).batchProgress
        assertEquals(stopped, resumed)
        assertEquals(1f, state(PhotoStatus.COMPLETED, PhotoStatus.FAILED).batchProgress.fraction, 0f)
        assertEquals(0f, PhotoDemoState().batchProgress.fraction, 0f)
    }
}
