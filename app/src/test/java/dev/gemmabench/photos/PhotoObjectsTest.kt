package dev.gemmabench.photos

import org.junit.Assert.*
import org.junit.Test

class PhotoObjectsTest {
    @Test fun parsesSeparateInstancesAndEdgeCoordinates() {
        val objects = PhotoObjects.parse("""{"objects":[{"name":"cat","box":[0,0,500,1000]},{"name":"cat","box":[501,20,1000,999]}]}""")
        assertEquals(2, objects.size)
        assertEquals("cat", objects[0].name)
        assertEquals(0f, objects[0].left)
        assertEquals(1000f, objects[1].right)
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
