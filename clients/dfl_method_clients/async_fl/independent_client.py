from clients.client import Client


class IndependentClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    def aggregate(self):
        pass

    def set_init_model(self, model):
        self.model = deepcopy(model)

    def train(self):
        self._local_train()

    def send_model(self) -> Dict:
        return {}