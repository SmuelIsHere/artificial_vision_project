from torch.utils.data import Sampler
import numpy as np

class CustomRandomSampler(Sampler):

    def __init__(self, dataset, replacement=False, batch_size=None):
        self.data_source = dataset
        self.replacement = replacement
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))

        number_iteration = len(indices) // self.batch_size

        if not self.replacement:
            indices = np.random.permutation(indices).tolist()

        batch = []
        index_seen = set()
        deque = []
        iteration = 1

        for idx in indices:
            if len(deque) > 0:
                t = min(self.batch_size - len(batch), len(deque))
                batch.extend(deque[:t])
                del deque[:t]
                if len(batch) == self.batch_size:
                    deque.append(idx)

            if len(batch) < self.batch_size:
                batch.append(idx)

            if len(batch) == self.batch_size:

                for batch_sample_index, sample_index in enumerate(batch):
                    _, labels = self.data_source[sample_index]

                    for label_index, l in enumerate(labels):

                        if l != -1:
                            index_seen.add(label_index)
                        elif l == -1 and label_index not in index_seen:

                            deque.append(sample_index)

                            
                            while(True):
                                index_new_sample = np.random.choice(len(self.data_source))
                                _, labels = self.data_source[index_new_sample]
                             
                                if labels[label_index] != -1:
                                    batch[batch_sample_index] = index_new_sample
                                    break

                yield batch

                if iteration == number_iteration:
                  break

                iteration += 1
                index_seen = set()
                batch = []

