import json
from .wdc_dataset import load_dataset, create_taxonomy


def get_statistics_wdc(dir_path):
    dataset_dict = load_dataset(dir_path)
    taxonomy = create_taxonomy(dataset_dict)
    stats = {}
    
    # Calculate taxonomy statistics
    stats['taxonomy'] = {
        'num_categories': len(taxonomy),
        'num_attributes': len({attribute for attributes in taxonomy.values() for attribute in attributes}),
        'num_category_attribute_pairs': sum(len(attributes) for attributes in taxonomy.values()),
        'num_category_attribute_value_tuples': sum(len(values) for attributes in taxonomy.values() for values in attributes.values())
    }

    # Calculate dataset statistics
    for split, df in dataset_dict.items():
        stats[split] = {
            'num_product': len(df),
            'num_pair': sum(len(row) for row in df['target_scores']),
            'num_pair_null': sum(list(row.values()).count({"n/a": 1}) for row in df['target_scores']),
        }

    return stats


if __name__ == "__main__":
    stats = get_statistics_wdc("../data/wdc_normalized")
    print(json.dumps(stats, indent=4))
