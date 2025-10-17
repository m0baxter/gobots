class LossAccumulator:
    def __init__(self):
        self.num_examples = 0
        self.running_total_loss = 0
        self.running_mtp_loss = 0
        self.running_auxiliary_loss = 0
        self.running_z_loss = 0

    def update(
        self,
        batch_total_loss: float,
        batch_mtp_loss: float,
        batch_axiliary_loss: float,
        batch_z_loss: float,
        batch_size: int,
    ):
        self.num_examples += batch_size
        self.running_total_loss += batch_total_loss * batch_size
        self.running_mtp_loss += batch_mtp_loss * batch_size
        self.running_auxiliary_loss += batch_axiliary_loss * batch_size
        self.running_z_loss += batch_z_loss * batch_size

    def compute(self):
        final_main_loss = (
            self.running_total_loss
            - (
                self.running_mtp_loss
                + self.running_auxiliary_loss
                + self.running_z_loss
            )
        ) / self.num_examples
        final_mtp_loss = self.running_mtp_loss / self.num_examples
        final_auxiliary_loss = self.running_auxiliary_loss / self.num_examples
        final_z_loss = self.running_z_loss / self.num_examples

        self.num_examples = 0
        self.running_total_loss = 0
        self.running_mtp_loss = 0
        self.running_auxiliary_loss = 0
        self.running_z_loss = 0

        return final_main_loss, final_mtp_loss, final_auxiliary_loss, final_z_loss
