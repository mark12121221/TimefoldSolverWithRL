package com.example.solver;

public class Task {

    private String id;
    private String requiredSkill;
    private int duration;
    private int priority;
    private boolean mandatory;

    // Default constructor for Jackson
    public Task() {}

    public Task(String id, String requiredSkill, int duration, int priority, boolean mandatory) {
        this.id = id;
        this.requiredSkill = requiredSkill;
        this.duration = duration;
        this.priority = priority;
        this.mandatory = mandatory;
    }

    // Getters and setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getRequiredSkill() { return requiredSkill; }
    public void setRequiredSkill(String requiredSkill) { this.requiredSkill = requiredSkill; }

    public int getDuration() { return duration; }
    public void setDuration(int duration) { this.duration = duration; }

    public int getPriority() { return priority; }
    public void setPriority(int priority) { this.priority = priority; }

    public boolean isMandatory() { return mandatory; }
    public void setMandatory(boolean mandatory) { this.mandatory = mandatory; }

    @Override
    public String toString() {
        return "Task{" +
                "id='" + id + '\'' +
                ", requiredSkill='" + requiredSkill + '\'' +
                ", duration=" + duration +
                ", priority=" + priority +
                ", mandatory=" + mandatory +
                '}';
    }
}