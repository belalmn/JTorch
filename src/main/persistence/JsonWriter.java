// Adapted from UBC CPSC 210 - JsonSerializationDemo

package persistence;

import org.json.JSONObject;

import java.io.*;

// Represents a reader that reads neural network from JSON data on file
public class JsonWriter {

    // EFFECTS: Constructs a writer to write to destination file
    public JsonWriter(String destination) {
        
    }

    // MODIFIES: this
    // EFFECTS: Opens the writer
    public void open() throws FileNotFoundException {

    }

    // MODIFIES: nn
    // EFFECTS: Writes JSON representation of a Writable object to file
    public void write(Writable writable) {

    }

    // EFFECTS: Closes the writer
    public void close() {

    }

    // EFFECTS: Writes string to file
    private void saveToFile(String json) {

    }
}
