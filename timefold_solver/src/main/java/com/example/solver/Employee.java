package com.example.solver;

import java.util.List;

public class Employee {

    private String id;
    private List<String> skills;
    private int availableCapacity;

    // Default constructor for Jackson
    public Employee() {}

    public Employee(String id, List<String> skills, int availableCapacity) {
        this.id = id;
        this.skills = skills;
        this.availableCapacity = availableCapacity;
    }

    // Getters and setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public List<String> getSkills() { return skills; }
    public void setSkills(List<String> skills) { this.skills = skills; }

    public int getAvailableCapacity() { return availableCapacity; }
    public void setAvailableCapacity(int availableCapacity) { this.availableCapacity = availableCapacity; }

    @Override
    public String toString() {
        return "Employee{" +
                "id='" + id + '\'' +
                ", skills=" + skills +
                ", availableCapacity=" + availableCapacity +
                '}';
    }
}