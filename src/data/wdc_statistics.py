import json
from .wdc_dataset import load_dataset, create_taxonomy


MEASUREMENT_ATTRIBUTES = ['Width', 'Height', 'Depth', 'Length']


def get_statistics_wdc(dir_path):
    dataset_dict = load_dataset(dir_path)
    taxonomy = create_taxonomy(dataset_dict)
    stats = {}
    
    # Calculate taxonomy statistics
    unique_attributes = {attribute for attributes in taxonomy.values() for attribute in attributes}
    all_pairs = sum(len(attributes) for attributes in taxonomy.values())
    all_tuples = sum(len(values) for attributes in taxonomy.values() for values in attributes.values())
    
    stats['taxonomy'] = {
        'num_categories': len(taxonomy),
        'num_attributes': len(unique_attributes),
        'num_category_attribute_pairs': all_pairs,
        'num_category_attribute_value_tuples': all_tuples
    }

    # Calculate dataset statistics
    for split, df in dataset_dict.items():
        stats[split] = {
            'num_product': len(df),
            'num_pair': sum(len(row) for row in df['target_scores']),
            'num_pair_null': sum(list(row.values()).count({"n/a": 1}) for row in df['target_scores']),
        }
        
        # Add filtered statistics (excluding measurement attributes)
        filtered_products = []
        for _, row in df.iterrows():
            # Filter out measurement attributes
            filtered_dict = {k: v for k, v in row['target_scores'].items() if k not in MEASUREMENT_ATTRIBUTES}
            if filtered_dict:
                filtered_products.append(filtered_dict)
        
        stats[f'{split}_filtered'] = {
            'num_product': len(filtered_products),
            'num_pair': sum(len(row) for row in filtered_products),
            'num_pair_null': sum(list(row.values()).count({"n/a": 1}) for row in filtered_products),
        }

    return stats


if __name__ == "__main__":
    stats = get_statistics_wdc("../data/wdc_normalized")
    print(json.dumps(stats, indent=4))
