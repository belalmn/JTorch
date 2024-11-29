package model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Calendar;
import java.util.Date;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the Event class
 */
public class EventTest {
	private Event e;
	private Date d;
	
	@BeforeEach
	public void runBefore() {
		e = new Event("Layer x added to network");
		d = Calendar.getInstance().getTime();
	}
	
	@Test
	public void testEvent() {
		assertEquals("Layer x added to network", e.getDescription());
        assertEquals(d.getTime() / 1000, e.getDate().getTime() / 1000);
	}

	@Test
	public void testToString() {
		assertEquals(d.toString() + "\n" + "Layer x added to network", e.toString());
	}

    @Test
    public void testHashCode() {
        Event e2 = new Event("Layer x added to network");
        assertEquals(e.hashCode(), e2.hashCode());

        Event e3 = new Event("Layer y added to network");
        assertNotEquals(e.hashCode(), e3.hashCode());
    }

    @Test
    public void testEquals() {
        Event e2 = new Event("Layer x added to network");
        assertTrue(e.equals(e2));

        Event e3 = new Event("Layer y added to network");
        assertNotEquals(e, e3);
    }

    @Test
    public void testEqualsNull() {
        assertNotEquals(e, null);
    }

    @Test
    public void testEqualsDifferentClass() {
        assertNotEquals(e, new Object());
    }
}
