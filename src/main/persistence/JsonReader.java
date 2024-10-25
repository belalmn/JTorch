package persistence;

import org.json.JSONObject;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

// Represents a reader that reads neural network from JSON data stored in file
public class JsonReader {
    private String source;

    // MODIFIES: this
    // EFFECTS: Constructs a reader to read from source file
    public JsonReader(String source) {
        this.source = source;
    }

    // MODIFIES: this
    // EFFECTS: Reads JSON from file and returns it
    public JSONObject readJson() throws IOException {
        String jsonData = readFile(source);
        return new JSONObject(jsonData);
    }

    // MODIFIES: this
    // EFFECTS: Reads source file as string and returns it
    private String readFile(String source) throws IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(source));
        return new String(encoded, StandardCharsets.UTF_8);
    }
}
