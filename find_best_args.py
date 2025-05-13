import subprocess
import itertools
import time
import re

MLP_SCRIPT_PATH = "mlp.py"
TRAIN_PATH = "datasets/Training.csv"
VALID_PATH = "datasets/Validation.csv"
BEST_N_RESULTS = 5
EPOCHS = "100"

learning_rates = [0.01, 0.001]
batch_sizes = [16, 32]
layer_configs = [
    [8, 8],
    [24, 24],
]

def parse_output(output_text):
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    last_val_loss = float('inf')
    last_val_f1 = 0.0
    
    epoch_regex = re.compile(r"epoch \d+/\d+ - loss: [\d.]+ - val_loss: ([\d.]+) - val_f1: ([\d.]+)")
    
    lines = output_text.splitlines()
    
    for line in lines:
        match = epoch_regex.search(line)
        if match:
            current_val_loss = float(match.group(1))
            current_val_f1 = float(match.group(2))
            
            last_val_loss = current_val_loss
            last_val_f1 = current_val_f1
            
            if current_val_loss < best_val_loss:
                 best_val_loss = current_val_loss
                 best_val_f1 = current_val_f1
            elif current_val_loss == best_val_loss:
                 if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1

    if best_val_loss == float('inf'):
        return None, None

    return last_val_loss, last_val_f1


def main_tuner():
    start_time = time.time()

    param_combinations = list(itertools.product(learning_rates, batch_sizes, layer_configs))
    total_combinations = len(param_combinations)

    results = []

    for i, (lr, bs, layers) in enumerate(param_combinations):
        layer_args = [str(unit) for unit in layers]
        
        command = [
            "python3", MLP_SCRIPT_PATH,
            "--mode", "train",
            "--train", TRAIN_PATH,
            "--valid", VALID_PATH,
            "--learning_rate", str(lr),
            "--batch_size", str(bs),
            "--epochs", EPOCHS,
            "--layer"
        ] + layer_args

        print(f"Running ({i+1}/{total_combinations}): lr={lr}, batch_size={bs}, layers={layers}")
        process = subprocess.run(command, capture_output=True, text=True, timeout=600)
        
        val_loss, val_f1 = parse_output(process.stdout)
        print(f"  Completed: val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")

        results.append({
            "learning_rate": lr,
            "batch_size": bs,
            "layers": layers,
            "val_loss": val_loss,
            "val_f1": val_f1
        })

    end_time = time.time()
    total_time_seconds = end_time - start_time
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)

    print("\n--- Summary ---")
    print(f"Execution time: {minutes} minutes and {seconds} seconds")

    successful_results = [r for r in results if r['val_loss'] is not None and r['val_f1'] is not None]

    best_by_loss = sorted(successful_results, key=lambda x: x['val_loss'])
    print(f"\nTop {BEST_N_RESULTS} configurations by Validation Loss (lower is better):")
    for i, res in enumerate(best_by_loss[:BEST_N_RESULTS]):
        print(f"  {i+1}. Loss: {res['val_loss']:.4f}, F1: {res['val_f1']:.4f} | Params: lr={res['learning_rate']}, bs={res['batch_size']}, layers={res['layers']}")

    best_by_f1 = sorted(successful_results, key=lambda x: x['val_f1'], reverse=True)
    print(f"\nTop {BEST_N_RESULTS} configurations by Validation F1 Score (higher is better):")
    for i, res in enumerate(best_by_f1[:BEST_N_RESULTS]):
        print(f"  {i+1}. F1: {res['val_f1']:.4f}, Loss: {res['val_loss']:.4f} | Params: lr={res['learning_rate']}, bs={res['batch_size']}, layers={res['layers']}")

if __name__ == "__main__":
    main_tuner()