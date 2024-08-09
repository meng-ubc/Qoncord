class QuantumJob:
    def __init__(
        self,
        job_id,
        time_per_job=5,
        device_id=None,
        runtime=False,
        circuits_per_runtime=100,
        runtime_cooldown=2,
        qoncord=False,
    ):
        # Job parameters
        self.job_id = job_id
        self.device_id = device_id
        self.time_per_job = time_per_job

        # Runtime job has multiple jobs
        self.runtime = runtime
        if runtime:
            self.remaining_jobs = circuits_per_runtime
        else:
            self.remaining_jobs = 1

        self.qoncord = qoncord
        self.device_switches = [
            int(2 * self.remaining_jobs / 3),
            int(self.remaining_jobs / 3),
        ]
        self.qoncord_devices = [self.device_id - 3, self.device_id - 6]

        self.circuits_reduction_rate = 1

        self.runtime_cooldown = runtime_cooldown
        self.cooldown_remaining = 0
        self.job_runing = False
        self.job_remaining_time = 0

        self.is_completed = False
        self.start_times = []
        self.end_times = []

        self.fidelity = None

        self.fluctuation = False

    def start(self, current_time):
        if self.cooldown_remaining > 0:
            return False

        self.remaining_jobs -= 1

        if self.qoncord and self.remaining_jobs in self.device_switches:
            next_device = self.qoncord_devices.pop(0)

            self.device_id = next_device

        self.start_times.append(current_time)
        self.end_times.append(current_time + self.time_per_job - 1)

        self.job_runing = True
        self.job_remaining_time = self.time_per_job
        return True

    def update_and_is_finished(self):
        if self.job_runing:
            self.job_remaining_time -= 1
            if self.job_remaining_time == 0:
                self.job_runing = False

                self.cooldown_remaining = self.runtime_cooldown

                if self.remaining_jobs == 0:
                    self.is_completed = True

                    # Job is completed
                    return True

        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        return False
