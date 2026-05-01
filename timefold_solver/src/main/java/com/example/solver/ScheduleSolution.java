package com.example.solver;

import ai.timefold.solver.core.api.domain.solution.PlanningEntityCollectionProperty;
import ai.timefold.solver.core.api.domain.solution.PlanningScore;
import ai.timefold.solver.core.api.domain.solution.PlanningSolution;
import ai.timefold.solver.core.api.domain.solution.ProblemFactCollectionProperty;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeProvider;
import ai.timefold.solver.core.api.score.buildin.hardsoft.HardSoftScore;

import java.util.List;

@PlanningSolution
public class ScheduleSolution {

    @ProblemFactCollectionProperty
    @ValueRangeProvider(id = "employeeRange")
    private List<Employee> employees;

    @ProblemFactCollectionProperty
    private List<Task> tasks;

    @PlanningEntityCollectionProperty
    private List<TaskAssignment> taskAssignments;

    @PlanningScore
    private HardSoftScore score;

    // Default constructor
    public ScheduleSolution() {
    }

    public ScheduleSolution(List<Employee> employees, List<Task> tasks, List<TaskAssignment> taskAssignments) {
        this.employees = employees;
        this.tasks = tasks;
        this.taskAssignments = taskAssignments;
    }

    // Getters and setters
    public List<Employee> getEmployees() {
        return employees;
    }

    public void setEmployees(List<Employee> employees) {
        this.employees = employees;
    }

    public List<Task> getTasks() {
        return tasks;
    }

    public void setTasks(List<Task> tasks) {
        this.tasks = tasks;
    }

    public List<TaskAssignment> getTaskAssignments() {
        return taskAssignments;
    }

    public void setTaskAssignments(List<TaskAssignment> taskAssignments) {
        this.taskAssignments = taskAssignments;
    }

    public HardSoftScore getScore() {
        return score;
    }

    public void setScore(HardSoftScore score) {
        this.score = score;
    }

    @Override
    public String toString() {
        return "ScheduleSolution{" +
                "employees=" + employees.size() +
                ", tasks=" + tasks.size() +
                ", taskAssignments=" + taskAssignments.size() +
                ", score=" + score +
                '}';
    }
}