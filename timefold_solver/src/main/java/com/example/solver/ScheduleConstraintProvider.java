package com.example.solver;

import ai.timefold.solver.core.api.score.buildin.hardsoft.HardSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.ConstraintProvider;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;

public class ScheduleConstraintProvider implements ConstraintProvider {

        private static final int MAX_TASKS_PER_EMPLOYEE = 5;

        @Override
        public Constraint[] defineConstraints(ConstraintFactory constraintFactory) {
                return new Constraint[] {
                                requiredSkill(constraintFactory),
                                capacityNotExceeded(constraintFactory),
                                allTasksAssigned(constraintFactory),
                                highPriorityTasksAssigned(constraintFactory),
                                maxTasksPerEmployee(constraintFactory),
                                mandatoryTasksAssigned(constraintFactory),
                                prioritizeHighPriorityTasks(constraintFactory),
                                balanceWorkload(constraintFactory)
                };
        }

        // Hard: A task cannot be assigned to an employee who lacks the required skill
        private Constraint requiredSkill(ConstraintFactory constraintFactory) {
                return constraintFactory.forEach(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() != null
                                                && !ta.getAssignedEmployee().getSkills()
                                                                .contains(ta.getTask().getRequiredSkill()))
                                .penalize(HardSoftScore.ONE_HARD, ta -> 1_000)
                                .asConstraint("Required skill");
        }

        // Hard: Exceeding capacity comes at a high cost;
        // The greater the overload, the faster the penalty increases
        private Constraint capacityNotExceeded(ConstraintFactory constraintFactory) {
                return constraintFactory.forEach(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() != null)
                                .groupBy(TaskAssignment::getAssignedEmployee,
                                                ConstraintCollectors.sum(ta -> ta.getTask().getDuration()))
                                .filter((employee, totalDuration) -> totalDuration > employee.getAvailableCapacity())
                                .penalize(HardSoftScore.ONE_HARD,
                                                (employee, totalDuration) -> {
                                                        int overload = totalDuration - employee.getAvailableCapacity();
                                                        return overload * overload * 50; // non-linear penalty
                                                })
                                .asConstraint("Capacity not exceeded");
        }

        private Constraint mandatoryTasksAssigned(ConstraintFactory constraintFactory) {
                return constraintFactory.forEachIncludingNullVars(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() == null && ta.getTask().isMandatory())
                                .penalize(HardSoftScore.ONE_HARD, ta -> 3_000)
                                .asConstraint("Mandatory tasks assigned");
        }

        // Hard: all tasks must be assigned
        // IMPORTANT: forEachIncludingNullVars(), otherwise null might not make it to
        // the stream
        private Constraint allTasksAssigned(ConstraintFactory constraintFactory) {
                return constraintFactory.forEachIncludingNullVars(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() == null)
                                .penalize(HardSoftScore.ONE_HARD, ta -> 500)
                                .asConstraint("All tasks assigned");
        }

        // Hard: high-priority tasks must be assigned
        // Even more strictly than regular unassigned tasks
        private Constraint highPriorityTasksAssigned(ConstraintFactory constraintFactory) {
                return constraintFactory.forEachIncludingNullVars(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() == null
                                                && ta.getTask().getPriority() >= 4)
                                .penalize(HardSoftScore.ONE_HARD,
                                                ta -> ta.getTask().getPriority() * 2_000)
                                .asConstraint("High priority tasks assigned");
        }

        // Hard: Limit on the number of tasks per employee
        // Makes exceeding the limit highly disadvantageous
        private Constraint maxTasksPerEmployee(ConstraintFactory constraintFactory) {
                return constraintFactory.forEach(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() != null)
                                .groupBy(TaskAssignment::getAssignedEmployee, ConstraintCollectors.count())
                                .filter((employee, taskCount) -> taskCount > MAX_TASKS_PER_EMPLOYEE)
                                .penalize(HardSoftScore.ONE_HARD,
                                                (employee, taskCount) -> {
                                                        int overflow = taskCount - MAX_TASKS_PER_EMPLOYEE;
                                                        return overflow * overflow * 200;
                                                })
                                .asConstraint("Max tasks per employee");
        }

        // Soft: reward for assigning priority tasks
        private Constraint prioritizeHighPriorityTasks(ConstraintFactory constraintFactory) {
                return constraintFactory.forEach(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() != null)
                                .reward(HardSoftScore.ONE_SOFT,
                                                ta -> ta.getTask().getPriority() * 2)
                                .asConstraint("Prioritize high priority");
        }

        // Soft: penalty for workload imbalance
        private Constraint balanceWorkload(ConstraintFactory constraintFactory) {
                return constraintFactory.forEach(TaskAssignment.class)
                                .filter(ta -> ta.getAssignedEmployee() != null)
                                .groupBy(TaskAssignment::getAssignedEmployee,
                                                ConstraintCollectors.sum(ta -> ta.getTask().getDuration()))
                                .penalize(HardSoftScore.ONE_SOFT,
                                                (employee, totalDuration) -> totalDuration * totalDuration)
                                .asConstraint("Balance workload");
        }
}