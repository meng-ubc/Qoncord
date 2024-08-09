class QuantumSimulation:
    def __init__(self, devices, jobs, timestep=1):
        self.devices = {device.device_id: device for device in devices}

        self.queued_jobs = {n: [] for n in range(len(devices))}

        for job in jobs:
            self.queued_jobs[job.device_id].append(job)

        self.unfinished_jobs = jobs

        self.current_time = 0
        self.timestep = timestep

    def run_simulation(self):
        while True:
            has_unfinished_jobs = False
            for device_id in self.queued_jobs:
                jobs_queue = self.queued_jobs[device_id]

                if len(jobs_queue) > 0:
                    has_unfinished_jobs = True

                    check_job = jobs_queue[0]
                    if not check_job.job_runing and check_job.device_id != device_id:
                        self.move_job_to_device(
                            check_job, device_id, check_job.device_id
                        )

                # if len(jobs_queue) > 1 and:

                if self.devices[device_id].is_idle():
                    for job in list(jobs_queue):
                        if job.start(self.current_time):
                            self.devices[device_id].assign_job(job)
                            break

            if not has_unfinished_jobs:
                break

            self.advance_time()

    def move_job_to_device(self, job, origional_device_id, other_device_id):
        # print(
        #     f"Moving job {job.job_id} from {origional_device_id} to {other_device_id}"
        # )
        try:
            if (
                job not in self.queued_jobs[origional_device_id]
                or job in self.queued_jobs[other_device_id]
            ):
                raise ValueError(
                    f"Job migration error from {origional_device_id} to {other_device_id}"
                )
        except KeyError:
            raise ValueError(
                f"Job migration error from {origional_device_id} to {other_device_id}"
            )

        # assert job in self.queued_jobs[origional_device_id]
        # assert job not in self.queued_jobs[other_device_id]

        self.queued_jobs[origional_device_id].remove(job)

        if self.devices[other_device_id].is_idle():
            # target device is idle, so we put the job in the first position
            self.queued_jobs[other_device_id].insert(0, job)
        else:
            # target device is busy, so we put the job in the second position
            self.queued_jobs[other_device_id].insert(1, job)

    def advance_time(self):
        # Advance devices
        for device in self.devices.values():
            device.advance_time()

        for job_list in self.queued_jobs.values():
            for job in job_list:
                if job.update_and_is_finished():
                    self.queued_jobs[job.device_id].remove(job)

        self.current_time += self.timestep
