package dev.gemmabench.people

import androidx.room.withTransaction
import dev.gemmabench.photos.DevicePhoto
import dev.gemmabench.photos.PhotoCacheDatabase
import dev.gemmabench.photos.PhotoCacheRequest
import java.util.UUID

class PeopleRepository(private val db: PhotoCacheDatabase) {
    private val dao get() = db.people()

    fun key(photo: DevicePhoto): String = PhotoCacheRequest.sha256(
        "${photo.uri}|${photo.sourceRevision}|${FaceEngine.VERSION}|${FaceEngine.DETECTOR_SHA}|${FaceEngine.RECOGNIZER_SHA}".toByteArray())

    suspend fun cached(photo: DevicePhoto): Boolean = dao.photo(photo.uri)?.let { it.cacheKey == key(photo) && it.status == "COMPLETED" } == true

    suspend fun save(photo: DevicePhoto, detected: List<DetectedFace>, accessible: Map<String, String>, policy: MatchPolicy) = db.withTransaction {
        val previous = dao.facesForPhoto(photo.uri).associateBy { it.id }
        val anchors = dao.anchors().filter { it.photoUri != photo.uri && it.modelVersion == FaceEngine.EMBEDDING_VERSION && accessible[it.photoUri] == it.revision }
            .associate { it.personId to FaceMatching.decode(it.embedding) }
        dao.deletePhotoFaces(photo.uri)
        for ((index, face) in detected.withIndex()) {
            val faceId = PhotoCacheRequest.sha256("${key(photo)}:$index".toByteArray())
            val old = previous[faceId]
            val blocked = dao.blocks(faceId).map { it.personId }.toSet()
            val personId = if (old?.manual == true) old.personId else face.embedding?.let { embedding ->
                FaceMatching.choose(anchors.filterKeys { it !in blocked }.map { (id, anchor) -> id to FaceMatching.cosine(embedding, anchor) }, policy)
                    ?: UUID.randomUUID().toString().also { dao.save(Person(it, anchorFaceId = faceId)) }
            }
            dao.save(PersonFace(faceId, photo.uri, personId, face.left, face.top, face.right, face.bottom,
                face.embedding?.let(FaceMatching::encode), face.quality, FaceEngine.EMBEDDING_VERSION, old?.manual ?: false))
        }
        dao.save(PeoplePhoto(photo.uri, photo.sourceRevision, key(photo), photo.name, "COMPLETED"))
        dao.deleteOrphanBlocks()
    }

    suspend fun failed(photo: DevicePhoto, error: String) {
        // Prior observations remain for diagnosis but are hidden until this revision completes.
        dao.save(PeoplePhoto(photo.uri, photo.sourceRevision, key(photo), photo.name, "FAILED", error))
    }

    suspend fun gallery(accessible: Map<String, String>): Pair<List<PersonGroup>, List<VisibleFace>> {
        val visible = dao.visibleFaces().filter { accessible[it.face.photoUri] == it.revision }.map { it.face }
        val grouped = visible.groupBy { it.personId }
        return dao.people().mapNotNull { p -> grouped[p.id]?.let { PersonGroup(p.id, p.name, it) } }
            .sortedWith(compareByDescending<PersonGroup> { it.photoCount }.thenBy { it.label }) to grouped[null].orEmpty()
    }

    suspend fun rename(id: String, name: String) { dao.rename(id, name.trim().take(120).ifEmpty { null }) }

    suspend fun merge(source: String, target: String) = db.withTransaction {
        require(source != target)
        val people = dao.people().associateBy { it.id }
        require(source in people && target in people) { "Person group is no longer available." }
        dao.transferBlocks(source, target)
        dao.deleteBlocksForPerson(source)
        dao.moveGroup(source, target)
        dao.alias(source, target)
        dao.unblockAssigned(target)
        if (people[target]!!.anchorFaceId == null) dao.save(people[target]!!.copy(anchorFaceId = people[source]!!.anchorFaceId))
    }

    suspend fun move(faceId: String, targetId: String?) = db.withTransaction {
        val face = dao.face(faceId) ?: error("Face is no longer available.")
        val people = dao.people().associateBy { it.id }
        val target = targetId ?: UUID.randomUUID().toString().also { dao.save(Person(it, anchorFaceId = faceId)) }
        require(targetId == null || target in people) { "Person group is no longer available." }
        face.personId?.takeIf { it != target }?.let { dao.block(PersonBlock(face.id, it)) }
        dao.unblock(face.id, target)
        dao.save(face.copy(personId = target, manual = true))
        face.personId?.let { old ->
            people[old]?.takeIf { it.anchorFaceId == faceId && old != target }?.let { person ->
                dao.save(person.copy(anchorFaceId = dao.replacementAnchor(old)))
            }
        }
    }

    suspend fun clear() = db.withTransaction { dao.clearBlocks(); dao.clearFaces(); dao.clearPhotos(); dao.clearPeople() }
}
