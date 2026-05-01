package com.example.solver;

import ai.timefold.solver.core.api.solver.Solver;
import ai.timefold.solver.core.api.solver.SolverFactory;
import ai.timefold.solver.core.config.solver.SolverConfig;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class SolverRunner {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java -jar timefold-solver.jar <input.json> <output.json>");
            System.exit(1);
        }

        String inputFile = args[0];
        String outputFile = args[1];

        try {
            // Read input
            InstanceData instanceData = objectMapper.readValue(new File(inputFile), InstanceData.class);

            // Build employee lookup for optional warm-start initialization
            Map<String, Employee> employeeMap = instanceData.getEmployees().stream()
                    .collect(Collectors.toMap(Employee::getId, e -> e));

            // Create task assignments, applying RL warm-start hints when present
            Map<String, String> initialAssignments = instanceData.getInitialAssignments();
            List<TaskAssignment> taskAssignments = instanceData.getTasks().stream()
                    .map(task -> {
                        TaskAssignment ta = new TaskAssignment(task);
                        if (initialAssignments != null) {
                            String empId = initialAssignments.get(task.getId());
                            if (empId != null) {
                                Employee emp = employeeMap.get(empId);
                                if (emp != null) {
                                    ta.setAssignedEmployee(emp);
                                }
                            }
                        }
                        return ta;
                    })
                    .collect(Collectors.toList());

            ScheduleSolution problem = new ScheduleSolution(instanceData.getEmployees(),
                    instanceData.getTasks(), taskAssignments);

            // Solve
            long startTime = System.currentTimeMillis();

            SolverFactory<ScheduleSolution> solverFactory = SolverFactory.create(new SolverConfig()
                    .withSolutionClass(ScheduleSolution.class)
                    .withEntityClasses(TaskAssignment.class)
                    .withConstraintProviderClass(ScheduleConstraintProvider.class)
                    .withTerminationSpentLimit(java.time.Duration.ofSeconds(30)));

            Solver<ScheduleSolution> solver = solverFactory.buildSolver();
            ScheduleSolution solution = solver.solve(problem);

            long runtimeMs = System.currentTimeMillis() - startTime;

            // Calculate results
            boolean isFeasible = solution.getScore().isFeasible();
            String score = solution.getScore().toString();
            int assignedTasks = (int) solution.getTaskAssignments().stream()
                    .filter(ta -> ta.getAssignedEmployee() != null)
                    .count();
            int unassignedTasks = solution.getTaskAssignments().size() - assignedTasks;

            // Create result
            ResultData result = new ResultData(isFeasible, score, runtimeMs, assignedTasks, unassignedTasks);

            // Write output
            objectMapper.writeValue(new File(outputFile), result);

            System.out.println("Solution completed. Feasible: " + isFeasible + ", Score: " + score + ", Runtime: "
                    + runtimeMs + "ms");

        } catch (IOException e) {
            System.err.println("Error processing files: " + e.getMessage());
            System.exit(1);
        }
    }

    // Helper classes for JSON
    public static class InstanceData {
        private List<Employee> employees;
        private List<Task> tasks;
        // Optional: RL-produced warm-start assignments (task_id -> employee_id).
        // When present, the solver starts local search from this initial solution.
        private Map<String, String> initialAssignments;

        public InstanceData() {
        }

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

        public Map<String, String> getInitialAssignments() {
            return initialAssignments;
        }

        public void setInitialAssignments(Map<String, String> initialAssignments) {
            this.initialAssignments = initialAssignments;
        }
    }

    public static class ResultData {
        private boolean isFeasible;
        private String score;
        private long runtimeMs;
        private int assignedTasks;
        private int unassignedTasks;

        public ResultData() {
        }

        public ResultData(boolean isFeasible, String score, long runtimeMs, int assignedTasks, int unassignedTasks) {
            this.isFeasible = isFeasible;
            this.score = score;
            this.runtimeMs = runtimeMs;
            this.assignedTasks = assignedTasks;
            this.unassignedTasks = unassignedTasks;
        }

        // Getters and setters
        public boolean isFeasible() {
            return isFeasible;
        }

        public void setFeasible(boolean isFeasible) {
            this.isFeasible = isFeasible;
        }

        public String getScore() {
            return score;
        }

        public void setScore(String score) {
            this.score = score;
        }

        public long getRuntimeMs() {
            return runtimeMs;
        }

        public void setRuntimeMs(long runtimeMs) {
            this.runtimeMs = runtimeMs;
        }

        public int getAssignedTasks() {
            return assignedTasks;
        }

        public void setAssignedTasks(int assignedTasks) {
            this.assignedTasks = assignedTasks;
        }

        public int getUnassignedTasks() {
            return unassignedTasks;
        }

        public void setUnassignedTasks(int unassignedTasks) {
            this.unassignedTasks = unassignedTasks;
        }
    }
}