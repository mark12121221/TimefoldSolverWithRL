package com.example.solver;

import ai.timefold.solver.core.api.domain.entity.PlanningEntity;
import ai.timefold.solver.core.api.domain.variable.PlanningVariable;

@PlanningEntity
public class TaskAssignment {

    private Task task;

    @PlanningVariable(valueRangeProviderRefs = "employeeRange", nullable = true)
    private Employee assignedEmployee;

    // Default constructor
    public TaskAssignment() {
    }

    public TaskAssignment(Task task) {
        this.task = task;
    }

    // Getters and setters
    public Task getTask() {
        return task;
    }

    public void setTask(Task task) {
        this.task = task;
    }

    public Employee getAssignedEmployee() {
        return assignedEmployee;
    }

    public void setAssignedEmployee(Employee assignedEmployee) {
        this.assignedEmployee = assignedEmployee;
    }

    @Override
    public String toString() {
        return "TaskAssignment{" +
                "task=" + task +
                ", assignedEmployee=" + (assignedEmployee != null ? assignedEmployee.getId() : "null") +
                '}';
    }
}