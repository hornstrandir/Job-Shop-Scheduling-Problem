import bisect
import datetime
import pickle
import random
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

# TODO:


class EnergyFlexibleJssEnv(gym.Env):
    def __init__(self, env_config=None):
        """
        This environment model the job shop scheduling problem as a single agent problem:
        -The actions correspond to a job allocation + one action for no allocation at this time step (NOPE action)
        -We keep a time with next possible time steps
        -Each time we allocate a job, the end of the job is added to the stack of time steps
        -If we don't have a legal action (i.e. we can't allocate a job),
        we automatically go to the next time step until we have a legal action
        -
        :param env_config: Ray dictionary of config parameter
        """
        instance_path = env_config["instance_path"]
        energy_data_path = env_config["energy_data_path"]
        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        ##################################################
        with open(energy_data_path, "rb") as file:
            self.ts_energy_prices = np.array(
                pickle.load(file, encoding="unicode_escape")
            )
        self.max_energy_price = np.amax(self.ts_energy_prices)
        self.current_energy_price = None
        self.penalty_weight = env_config["penalty_weight"]  # alpha
        ##################################################
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_op_jobs = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        self._total_energy_costs = 0
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        instance_file = open(instance_path, "r")
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                # matrix which store tuple of (machine, length of the job)
                self.instance_matrix = np.zeros(
                    (self.jobs, self.machines), dtype=(int, 2)
                )
                # contains all the time to complete jobs
                self.jobs_length = np.zeros(self.jobs, dtype=int)
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.machines
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2
                while i < len(split_data):
                    machine, time = int(split_data[i]), int(split_data[i + 1])
                    self.instance_matrix[job_nb][i // 2] = (machine, time)
                    self.max_time_op = max(self.max_time_op, time)
                    self.jobs_length[job_nb] += time
                    self.sum_op += time
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()
        self.power_consumption_machines = np.array(
            env_config["power_consumption_machines"][str(self.machines)]
        )
        self.max_power_consumption = np.max(self.power_consumption_machines)
        self.max_time_jobs = max(self.jobs_length)
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_matrix is not None
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 9), dtype=float
                ),
            }
        )

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self._total_energy_costs = 0
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs + 1, dtype=bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_op_jobs = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        self.current_energy_price = self.ts_energy_prices[0]
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.needed_machine_jobs[job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.jobs, 9), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float("inf")
                    for job in range(self.jobs):
                        if (
                            self.needed_machine_jobs[job] == machine
                            and self.legal_actions[job]
                        ):
                            if self.todo_op_jobs[job] == (self.machines - 1):
                                final_job.append(job)
                            else:
                                current_time_step_non_final = self.todo_op_jobs[
                                    job
                                ]
                                time_needed_legal = self.instance_matrix[job][
                                    current_time_step_non_final
                                ][1]
                                machine_needed_nextstep = self.instance_matrix[job][
                                    current_time_step_non_final + 1
                                ][0]
                                if (
                                    self.time_until_available_machine[
                                        machine_needed_nextstep
                                    ]
                                    == 0
                                ):
                                    min_non_final = min(
                                        min_non_final, time_needed_legal
                                    )
                                    non_final_job.append(job)
                    if len(non_final_job) > 0:
                        for job in final_job:
                            current_time_step_final = self.todo_op_jobs[job]
                            time_needed_legal = self.instance_matrix[job][
                                current_time_step_final
                            ][1]
                            if time_needed_legal > min_non_final:
                                self.legal_actions[job] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self):
        self.legal_actions[self.jobs] = False
        if (
            len(self.next_time_step) > 0
            and self.nb_machine_legal <= 3
            and self.nb_legal_actions <= 4
        ):
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_op_jobs[job]
                    machine_needed = self.instance_matrix[job][time_step][0]
                    time_needed = self.instance_matrix[job][time_step][1]
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            for job in range(self.jobs):
                if not self.legal_actions[job]:
                    if (
                        self.time_until_finish_current_op_jobs[job] > 0
                        and self.todo_op_jobs[job] + 1 < self.machines
                    ):
                        time_step = self.todo_op_jobs[job] + 1
                        time_needed = (
                            self.current_time_step
                            + self.time_until_finish_current_op_jobs[job]
                        )
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                    elif (
                        not self.action_illegal_no_op[job]
                        and self.todo_op_jobs[job] < self.machines
                    ):
                        time_step = self.todo_op_jobs[job]
                        machine_needed = self.instance_matrix[job][time_step][0]
                        time_needed = (
                            self.current_time_step
                            + self.time_until_available_machine[machine_needed]
                        )
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def step(self, action: int):
        reward = 0.0
        if action == self.jobs:
            self.nb_machine_legal = 0
            self.nb_legal_actions = 0
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    self.legal_actions[job] = False
                    needed_machine = self.needed_machine_jobs[job]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][job] = True
                    self.action_illegal_no_op[job] = True
            while self.nb_machine_legal == 0:
                reward -= self.increase_time_step()
            scaled_reward = self._reward_scaler(reward, energy_penalty=0)
            self._prioritization_non_final()
            self._check_no_op()
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )
        else:
            current_time_step_job = self.todo_op_jobs[action]
            machine_needed = self.needed_machine_jobs[action]
            time_needed = self.instance_matrix[action][current_time_step_job][1]
            energy_penalty = self._calculate_energy_penalty(action, time_needed)
            reward += time_needed
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_finish_current_op_jobs[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_jobs.insert(index, action)
            self.solution[action][current_time_step_job] = self.current_time_step
            # Set actions that need the currently allocated machine as illegal.
            for job in range(self.jobs):
                if (
                    self.needed_machine_jobs[job] == machine_needed
                    and self.legal_actions[job]
                ):
                    self.legal_actions[job] = False
                    self.nb_legal_actions -= 1
            self.nb_machine_legal -= 1
            self.machine_legal[machine_needed] = False
            for job in range(self.jobs):
                if self.illegal_actions[machine_needed][job]:
                    self.action_illegal_no_op[job] = False
                    self.illegal_actions[machine_needed][job] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                reward -= self.increase_time_step()
            self._prioritization_non_final()
            self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(reward, energy_penalty)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )

    def _update_power_observations(self):
        """
        Must be called after increasing the timestep since the calculations
        depend on the next states repr.
        """
        for job in range(self.jobs):
            ## TODO: Dont know if 0 is a sophisticated solution
            if self.todo_op_jobs[job] == self.machines:
                self.state[job][7] = 1.0
                self.state[job][8] = 1.0
            else:
                needed_machine = self.needed_machine_jobs[job]
                self.state[job][7] = (
                    self.power_consumption_machines[needed_machine]
                    / self.max_power_consumption
                )
                duration = self.instance_matrix[job, self.todo_op_jobs[job]][1]
                # since we are only interested in observations of legal actions,
                # we take the avg of the time when the action can be selected.
                avg_price = np.average(
                    self.ts_energy_prices[
                        self.current_time_step : self.current_time_step + duration
                    ]
                )
                self.state[job][8] = avg_price / self.max_energy_price

    # TODO: energy penalty can contain total energy costs
    def _calculate_energy_penalty(self, action: int, processing_time: int):
        """
        Calculate the energy penalty. The penalty is scaled by the max_price.

        penalty = avg(price_vector) * power_consumption_machine / 60 / max_price
        """
        avg_price = np.average(
            self.ts_energy_prices[
                self.current_time_step : self.current_time_step + processing_time
            ]
        )
        power_consumption = self.power_consumption_machines[
            self.needed_machine_jobs[action]
        ]
        # using ratio avg_price/max_energy_price thus unitless
        return avg_price / self.max_energy_price * power_consumption / self.max_power_consumption

    # TODO: Impact of ignoring energy_reward for noop.
    def _reward_scaler(self, reward: float, energy_penalty: float = 0):
        """
        Calculate the scaled reward consisting of the regular unscaled reward
        and the scaled energy_penalty.
        """
        if energy_penalty == 0:
            return reward / self.max_time_op
        return (1 - self.penalty_weight) * (
            reward / self.max_time_op
        ) - self.penalty_weight * energy_penalty

    def _update_total_energy_costs(self):
        power_consumption = self.power_consumption_machines.copy()
        # Legal machines are idle. Thus they don't consume energy.
        power_consumption[self.machine_legal] = 0
        self._total_energy_costs += np.sum(
            power_consumption * self.ts_energy_prices[self.current_time_step]
        )

    def increase_time_step(self):
        """
        The heart of the logic his here, we need to increase every
        counter when we have a nope action called
        and return the time elapsed
        :return: time elapsed
        """
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                # Note: performed_op_job is the delta_t to the next time step if the op is not already finished before.
                performed_op_job = min(difference, was_left_time)
                self.time_until_finish_current_op_jobs[job] = max(
                    0, self.time_until_finish_current_op_jobs[job] - difference
                )
                self.state[job][1] = (
                    self.time_until_finish_current_op_jobs[job] / self.max_time_op
                )
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = (
                    self.total_perform_op_time_jobs[job] / self.max_time_jobs
                )
                if self.time_until_finish_current_op_jobs[job] == 0:
                    self.total_idle_time_jobs[job] += difference - was_left_time
                    self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                    self.idle_time_jobs_last_op[job] = difference - was_left_time
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                    self.todo_op_jobs[job] += 1
                    self.state[job][2] = self.todo_op_jobs[job] / self.machines
                    if self.todo_op_jobs[job] < self.machines:
                        self.needed_machine_jobs[job] = self.instance_matrix[job][
                            self.todo_op_jobs[job]
                        ][0]
                        self.state[job][4] = (
                            max(
                                0,
                                self.time_until_available_machine[
                                    self.needed_machine_jobs[job]
                                ]
                                - difference,
                            )
                            / self.max_time_op
                        )
                    else:
                        self.needed_machine_jobs[job] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a
                        # good candidate)
                        self.state[job][
                            4
                        ] = 1.0  # a5: required time until the machine for the next op is free, scaled
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            elif self.todo_op_jobs[job] < self.machines:
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - difference
            )
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if (
                        self.needed_machine_jobs[job] == machine
                        and not self.legal_actions[job]
                        and not self.illegal_actions[machine][job]
                    ):
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        self._update_power_observations()
        self._update_total_energy_costs()
        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False

    def render(self, mode="human"):
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][i][0]
                )
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            fig.update_yaxes(
                autorange="reversed"
            )  # otherwise tasks are listed from the bottom up
        return fig

    @property
    def total_energy_costs(self):
        total_energy_costs = 0
        for job in range(self.jobs):
            for op in range(self.machines):
                op_start_time = self.solution[job][op]
                machine = self.instance_matrix[job][op][0]
                op_duration = self.instance_matrix[job][op][1]
                avg_price = np.average(
                    self.ts_energy_prices[op_start_time:op_start_time+op_duration]
                )
                total_energy_costs += avg_price*self.power_consumption_machines[machine]*op_duration
        return total_energy_costs