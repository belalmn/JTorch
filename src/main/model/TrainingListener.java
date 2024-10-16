package model;

public interface TrainingListener {
    
    // Called at the end of each training epoch.
    void onEpochEnd(int epoch, int totalEpochs, double loss);
}
