// Adapted from UBC CPSC 210 - JsonSerializationDemo

package persistence;

import org.json.JSONObject;

import java.io.*;

// Represents a reader that reads neural network from JSON data on file
public class JsonWriter {
    private PrintWriter writer;
    private String destination;

    // EFFECTS: Constructs a writer to write to destination file
    public JsonWriter(String destination) {
        this.destination = destination;
    }

    // MODIFIES: this
    // EFFECTS: Opens the writer
    public void open() throws FileNotFoundException {
        writer = new PrintWriter(new File(destination));
    }

    // MODIFIES: nn
    // EFFECTS: Writes JSON representation of a Writable object to file
    public void write(Writable writable) {
        JSONObject json = writable.toJson();
        saveToFile(json.toString(4)); // Indent with 4 spaces
    }

    // EFFECTS: Closes the writer
    public void close() {
        writer.close();
    }

    // EFFECTS: Writes string to file
    private void saveToFile(String json) {
        writer.print(json);
    }
}
