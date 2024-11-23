package ui.gui;

/**
 * Interface for classes that listen for changes in the tensor list.
 */
public interface TensorChangeListener {
    // EFFECTS: Executes a task when the tensor list is changed.
    void onTensorListChanged();
}