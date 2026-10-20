package dev.gemmabench.photos

import org.junit.Assert.*
import org.junit.Test

class PhotoObjectsTest {
    @Test fun truncatedAndMalformedJsonProduceActionableErrorsAndNeverPartialDetections() {
        listOf("", "{\"objects\":[", """{"objects":[{"name":"cat","box":[0,0,100,100]}""",
            "{\"objects\":not_json}").forEach { text ->
            val error = assertThrows(PhotoResponseException::class.java) { PhotoObjects.parse(text) }
            assertTrue(error.message!!.contains("response", ignoreCase = true))
        }
    }

    @Test fun rejectsTrailingContentAndDoesNotAcceptCommentsAsJson() {
        listOf("{\"objects\":[]} trailing", "{\"objects\":[]}{}", "{/*comment*/\"objects\":[]}").forEach {
            assertThrows(IllegalArgumentException::class.java) { PhotoObjects.parse(it) }
        }
    }

    @Test fun parsesSeparateInstancesAndEdgeCoordinates() {
        val objects = PhotoObjects.parse("""{"objects":[{"name":"cat","box":[0,0,1000,500]},{"name":"cat","box":[20,501,999,1000]}]}""")
        assertEquals(2, objects.size)
        assertEquals("cat", objects[0].name)
        assertEquals(0f, objects[0].left)
        assertEquals(1000f, objects[1].right)
    }

    @Test fun convertsGemmaYxyxToDisplayXyxyWithoutRotatingOrFlipping() {
        val objects = PhotoObjects.parse("""{"objects":[{"name":"person","box":[394,264,874,585]},{"name":"car","box":[313,313,715,794]}]}""")
        assertEquals(ObjectBox("person", left = 264f, top = 394f, right = 585f, bottom = 874f), objects[0])
        assertEquals(ObjectBox("car", left = 313f, top = 313f, right = 794f, bottom = 715f), objects[1])
        // On an 800 x 1200 portrait preview, each axis uses its own dimension.
        val person = objects[0]
        assertEquals(211.2f, person.left / 1000f * 800f, 0.01f)
        assertEquals(472.8f, person.top / 1000f * 1200f, 0.01f)
        assertEquals(468f, person.right / 1000f * 800f, 0.01f)
        assertEquals(1048.8f, person.bottom / 1000f * 1200f, 0.01f)
    }

    @Test fun acceptsFencedJsonAndExplicitlyEmptyScenes() {
        assertTrue(PhotoObjects.parse("```json\n{\"objects\":[]}\n```").isEmpty())
    }

    @Test fun rejectsBadCoordinatesRatherThanDrawingFalseBoxes() {
        listOf("[-1,0,100,100]", "[0,0,1001,100]", "[100,0,0,100]", "[0,10,100,10]",
            "[0,0,100]", "[0,0,\"100\",100]", "[0,0,null,100]").forEach { box ->
            assertThrows("Box $box", IllegalArgumentException::class.java) {
                PhotoObjects.parse("""{"objects":[{"name":"cat","box":$box}]}""")
            }
        }
    }

    @Test fun rejectsMissingOrInventedSchemas() {
        listOf("{}", "[]", "{\"objects\":null}", "{\"objects\":[{}]}",
            "{\"objects\":[{\"name\":\"\",\"box\":[0,0,100,100]}]}").forEach { json ->
            assertThrows(IllegalArgumentException::class.java) { PhotoObjects.parse(json) }
        }
    }
}
