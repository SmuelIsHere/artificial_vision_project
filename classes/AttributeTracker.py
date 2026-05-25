from collections import defaultdict, deque

class AttributeTracker:

    def __init__(self, max_entries=10):
        self.data = defaultdict(lambda: {
            'gender': deque(maxlen=max_entries),  
            'hat': deque(maxlen=max_entries),    
            'bag': deque(maxlen=max_entries)  
        })
        self.max_entries = max_entries

    def add_probabilities(self, person_id, gender_prob, hat_prob, bag_prob):
        """
        aggiunge la probabilità al dizionario per l'id specificato
        """
        self.data[person_id]['gender'].append(gender_prob)
        self.data[person_id]['hat'].append(hat_prob)
        self.data[person_id]['bag'].append(bag_prob)

    def get_average_probabilities(self, person_id):
        """
        calcola le medie delle probabilità per l'id specificato
        """
        if person_id not in self.data:
            return None
        
        return {
            'gender_avg': (sum(self.data[person_id]['gender'])-self.data[person_id]['gender'][0])/(len(self.data[person_id]['gender'])-1),
            'hat_avg': (sum(self.data[person_id]['hat'])-self.data[person_id]['hat'][0]) / (len(self.data[person_id]['hat'])-1),
            'bag_avg': (sum(self.data[person_id]['bag'])-self.data[person_id]['bag'][0]) / (len(self.data[person_id]['bag'])-1)
        }

    def clear_data(self, person_id):
        """
        rimuove i dati associati a un ID specifico.
        """
        if person_id in self.data:
            del self.data[person_id]
    def get_probabilities_count(self, person_id):
        """
        ritorna il numero di probabilità associate a un ID specifico
        """
        if person_id not in self.data:
            return 0
        # lunghezza di una delle deque, tanto lunghe uguale
        return len(self.data[person_id]['gender'])