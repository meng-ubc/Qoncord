import random

class QuantumDevice:
    def __init__(
        self,
        device_id,
        fidelity,
    ):
        self.device_id = device_id
        self.current_job = None
        self.time_remaining = 0
        self.fidelity = fidelity

    def assign_job(self, job):
        assert self.is_idle()

        variance = 0.1

        if random.random() < 0.1:  # 10% chance to increase variance to 0.3
            variance = 0.2

        # Generate a random fidelity within ± variance of the reference fidelity
        random_fidelity = random.uniform(
            self.fidelity - variance, self.fidelity + variance
        )

        # If base fidelity > 0.5, 10% chance to reduce the job fidelity by 0.1 (multi-programming effect)
        if self.fidelity > 0.5 and random.random() < 0.5:
            random_fidelity -= random.uniform(0, 0.1)

        # Ensuring fidelity is between 0 and 1
        random_fidelity = max(0, min(random_fidelity, 1))

        self.current_job = job
        self.time_remaining = job.time_per_job
        job.fidelity = random_fidelity

        return True

    def advance_time(self):

        if self.current_job and self.time_remaining > 0:
            self.time_remaining -= 1
            if self.time_remaining == 0:
                self.current_job.is_completed = True
                self.current_job = None

    def is_idle(self):
        return self.current_job is None
