def collate_fn(data):
    class_mapping = {'easy': 0, 'medium': 1, 'hard': 2}
    texts = [entry['text'] for entry in data]
    labels = [class_mapping[entry['difficulty']] for entry in data]
    return texts, labels
