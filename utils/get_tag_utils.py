def process_file(filename):
    data_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            key = parts[0]
            tags = [p.replace('<tag>', '').strip() for p in parts if p.startswith('<tag>')]
            attributes = [p.replace('<attribute>', '').strip() for p in parts if p.startswith('<attribute>')]
            sub_dict = {
                'tag': '-'.join(tags),
                'attribute': '-'.join(attributes)
            }
            data_dict[key] = sub_dict
    return data_dict

def construct_text(base_text, data):
    tag_probability_words = ["most likely", "probably", "perhaps"]
    attribute_probability_words = ["Likely", "Perhaps", "Could be"]
    
    tags = data['tag'].split('-')[:3]
    attributes = data['attribute'].split('-')[:1]
    
    # Constructing tag text
    tag_text = []
    for i, tag in enumerate(tags):
        tag_text.append(f"{tag_probability_words[i]} a {tag}")
    
    # Constructing attribute text
    attribute_text = []
    for i, attribute in enumerate(attributes):
        attribute_text.append(f"{attribute_probability_words[i]} {attribute}")

    final_text = base_text + ", " + ", ".join(tag_text) + ". " + ". ".join(attribute_text) + "."
    return final_text