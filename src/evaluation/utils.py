import csv
import pandas as pd
from src.evaluation.ndcg_evaluator import NDCGEvaluator

def load_golden_dataset_from_csv(path="data/fringe_golden_dataset.csv"):
    golden_dataset = []

    print(f"🔍 Loading golden dataset from: {path}")

    try:
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')

            # Debug: Zeige gefundene Spalten
            print(f"📋 Found columns: {list(reader.fieldnames)}")
            print(f"📋 Column representations: {[repr(col) for col in reader.fieldnames]}")

            # Prüfe auf unsichtbare Zeichen
            for col in reader.fieldnames:
                if 'Doc' in col and 'ID' in col:
                    print(f"✅ Found Doc IDs column: {repr(col)}")

            for row_num, row in enumerate(reader, 1):
                print(f"\n📝 Processing row {row_num}:")
                print(f"   Available keys: {list(row.keys())}")

                # Flexible Spalten-Suche
                query_key = None
                doc_ids_key = None

                for key in row.keys():
                    if 'query' in key.lower():
                        query_key = key
                    elif 'doc' in key.lower() and 'id' in key.lower():
                        doc_ids_key = key

                if not query_key:
                    query_key = list(row.keys())[0]  # First column
                if not doc_ids_key:
                    doc_ids_key = list(row.keys())[1]  # Second column

                print(f"   Using query key: {repr(query_key)}")
                print(f"   Using doc_ids key: {repr(doc_ids_key)}")

                try:
                    query = row[query_key].strip()
                    doc_ids_raw = row[' Doc IDs'].strip()

                    print(f"   Query: {query}")
                    print(f"   Doc IDs raw: {doc_ids_raw}")

                    if not query or not doc_ids_raw:
                        print("   ⚠️ Skipping empty row")
                        continue

                    doc_ids = [int(doc_id.strip()) for doc_id in doc_ids_raw.split(',') if doc_id.strip()]
                    expected_results = [{"episode_id": doc_id, "relevance": 1} for doc_id in doc_ids]

                    golden_dataset.append({
                        "query": query,
                        "expected_results": expected_results
                    })

                    print(f"   ✅ Successfully processed: {len(doc_ids)} doc IDs")

                except KeyError as e:
                    print(f"   ❌ KeyError: {e}")
                    print(f"   Available keys: {list(row.keys())}")
                    raise

    except Exception as e:
        print(f"❌ Error loading golden dataset: {e}")
        print(f"❌ Error type: {type(e)}")
        raise

    print(f"✅ Loaded {len(golden_dataset)} queries from golden dataset")
    return golden_dataset


def simple_load_golden_dataset_from_csv(path="data/fringe_golden_dataset.csv"):
    golden_dataset = []

    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')

        # Skip header
        header = next(reader)
        print(f"📋 Header: {header}")

        for row in reader:
            if len(row) < 2:
                continue

            query = row[0].strip()
            doc_ids_raw = row[1].strip()

            if not query or not doc_ids_raw:
                continue

            doc_ids = [doc_id.strip() for doc_id in doc_ids_raw.split(',') if doc_id.strip()]
            expected_results = [{"episode_id": doc_id, "relevance": 1} for doc_id in doc_ids]

            golden_dataset.append({
                "query": query,
                "expected_results": expected_results
            })

    print(f"✅ Loaded {len(golden_dataset)} queries from golden dataset")
    return golden_dataset


# Backup function - use this if the main one fails
def load_golden_dataset_from_csv_backup(path="data/fringe_golden_dataset.csv"):
   

    # Try approach 1: Original with better error handling
    try:
        return load_golden_dataset_from_csv(path)
    except Exception as e1:
        print(f"⚠️ Method 1 failed: {e1}")

        # Try approach 2: Simple CSV reader
        try:
            return simple_load_golden_dataset_from_csv(path)
        except Exception as e2:
            print(f"⚠️ Method 2 failed: {e2}")

            # Try approach 3: Different encoding
            try:
                golden_dataset = []
                with open(path, 'r', encoding='latin-1') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';')
                    next(reader)  # Skip header

                    for row in reader:
                        if len(row) >= 2:
                            query = row[0].strip()
                            doc_ids_raw = row[1].strip()
                            doc_ids = [doc_id.strip() for doc_id in doc_ids_raw.split(',') if doc_id.strip()]
                            expected_results = [{"episode_id": doc_id, "relevance": 1} for doc_id in doc_ids]
                            golden_dataset.append({"query": query, "expected_results": expected_results})

                print(f"✅ Loaded with latin-1 encoding: {len(golden_dataset)} queries")
                return golden_dataset

            except Exception as e3:
                print(f"❌ All methods failed: {e1}, {e2}, {e3}")
                raise e1  # Return original error



#def load_golden_dataset_from_csv(path="data/fringe_golden_dataset.csv"):
#    golden_dataset = []
#    with open(path, newline='', encoding='utf-8') as csvfile:
#        reader = csv.DictReader(csvfile, delimiter=';')
#        for row in reader:
#            query = row['Query'].strip()
#            doc_ids_raw = row['Doc IDs'].strip()
#
#            if not query or not doc_ids_raw:
#                continue  # Leere Zeile überspringen

 #           doc_ids = [doc_id.strip() for doc_id in doc_ids_raw.split(',') if doc_id.strip()]
#            expected_results = [{"episode_id": doc_id, "relevance": 1} for doc_id in doc_ids]
#
#            golden_dataset.append({
#                "query": query,
#                "expected_results": expected_results
#           })

#    return golden_dataset
