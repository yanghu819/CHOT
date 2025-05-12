import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
from transformers import AutoTokenizer # Added

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
os.environ['HF_HOME'] = '/ssdwork/huyang/r1' # Set HF cache if needed

# --- Define File Paths ---
# !! MODIFY THESE PATHS !!
base_dir = "/ssdwork/huyang/r1/simple_GRPO_debug/simple_grpo_v1_index_compare/" # ADJUST
right_indices_path = os.path.join(base_dir, "right.npy")
wrong_indices_path = os.path.join(base_dir, "wrong.npy")
delta_w_path = os.path.join(base_dir, "delta_w.npy")
# !! ADJUST TOKENIZER PATH !!
tokenizer_path = "/ssdwork/huyang/r1/simple_GRPO_debug/simple_grpo_v1/step_100"

# --- Output Directory ---
output_dir_png = os.path.join(base_dir, "visualizations_png_matplotlib_text") # New output dir

# --- Matplotlib Plotting Constants ---
FIG_WIDTH_INCHES = 18 # Increased width further for text
FIG_HEIGHT_INCHES = 10
FONT_SIZE = 8 # Smaller font for potentially longer text
H_PADDING_FACTOR = 1.4 # Multiplier for horizontal space based on char width
V_PADDING = 0.025 # Vertical padding between lines
TITLE_SPACE = 0.05 # Space reserved at top for titles

# --- Load Data ---
try:
    right_indices = np.load(right_indices_path, allow_pickle=True)
    wrong_indices = np.load(wrong_indices_path, allow_pickle=True)
    delta_w = np.load(delta_w_path, allow_pickle=True)
    logging.info(f"Data loaded. Samples: {len(right_indices)}, delta_w shape: {getattr(delta_w, 'shape', 'N/A')}")
except Exception as e:
    logging.error(f"Failed to load data: {e}. Exiting.", exc_info=True)
    exit()

# --- Load Tokenizer ---
tokenizer = None
logging.info(f"Attempting to load tokenizer from: {tokenizer_path}")
try:
    if os.path.isdir(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logging.info("Tokenizer loaded successfully.")
    else:
        logging.warning(f"Tokenizer path '{tokenizer_path}' does not exist or is not a directory.")
except Exception as e:
    logging.warning(f"Failed to load tokenizer from '{tokenizer_path}'. Error: {e}", exc_info=True)
if tokenizer is None:
     logging.warning("Proceeding without tokenizer - will display IDs only.")

# --- Prepare Output Directory ---
if not os.path.exists(output_dir_png):
    try:
        os.makedirs(output_dir_png)
        logging.info(f"Created directory: {output_dir_png}")
    except Exception as e:
        logging.error(f"Could not create output directory {output_dir_png}: {e}. Exiting.")
        exit()

# --- Helper Function to Decode Tokens Safely ---
def decode_token_safe(tokenizer, token_id):
    """Decodes token ID to string, handling errors and visual spacing."""
    if not tokenizer:
        return f"ID:{token_id}"
    try:
        # Decode preserving spaces, important for layout
        token_str = tokenizer.decode([token_id],
                                     skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

        # Replace common control/whitespace chars with visual symbols
        token_str = token_str.replace(' ', '␣') # Visual space
        token_str = token_str.replace('\n', '↵') # Visual newline
        token_str = token_str.replace('\t', '→') # Visual tab

        # Handle potential empty strings after decoding special tokens
        if not token_str:
             vocab = getattr(tokenizer, 'vocab', None) # Get vocab if possible
             if vocab:
                 inv_vocab = {v: k for k, v in vocab.items()}
                 raw_token = inv_vocab.get(token_id, f"[UNK:{token_id}]")
                 return f"[{raw_token.strip()}]" # Represent special tokens like [CLS], [PAD]
             else:
                 return f"[EMPTY:{token_id}]" # Fallback if vocab not easily accessible
        return token_str

    except Exception as e:
        logging.warning(f"Could not decode token ID {token_id}: {e}", exc_info=False) # Reduce verbosity
        return f"[ERR:{token_id}]"


# --- Helper Function to Plot Tokens (Updated for Text) ---
def plot_tokens_matplotlib_text(ax, tokens, start_y, color_info, fig_width_norm, tokenizer):
    """Plots decoded tokens on matplotlib axes with basic wrapping."""
    if not tokens:
         ax.text(0.01, start_y, "(None)", fontsize=FONT_SIZE, color='grey', va='top', ha='left')
         return start_y - V_PADDING

    # More refined (but still approximate) width estimation based on characters
    # Assumes monospace/near-monospace rendering for estimation
    # Convert points to inches, estimate char width in inches, normalize by figure width
    char_width_estimate_norm = (FONT_SIZE / 72) * 0.6 / FIG_WIDTH_INCHES # Approx width of one char in axes coords

    x_pos = 0.01 # Start with small left padding
    y_pos = start_y
    max_y_reached = start_y

    for idx, token_id in enumerate(tokens):
        # Decode token to get actual text
        token_text = decode_token_safe(tokenizer, token_id)

        # Determine text color and background color (same logic as before)
        text_color = 'black'
        bg_color = 'white'
        try:
            if color_info['type'] == 'fixed':
                text_color = color_info.get('color', 'black')
                bg_color = color_info.get('bgcolor', 'white')
            elif color_info['type'] == 'dynamic':
                bg_color = color_info.get('bgcolor', 'white')
                values = color_info.get('values', [])
                if idx < len(values):
                    value = values[idx]
                    norm = color_info.get('norm')
                    cmap = color_info.get('cmap')
                    if norm and cmap: text_color = cmap(norm(value))
                    elif values: text_color = cmap(0.5) # Single value case
                # Else: Mismatch or error, use default text_color
        except Exception as e:
             logging.warning(f"Error determining color for token '{token_text}' (ID {token_id}): {e}")
             text_color = 'red'

        # --- Wrapping Logic based on Estimated Text Width ---
        current_token_width_estimate = len(token_text) * char_width_estimate_norm * H_PADDING_FACTOR # Estimate + padding
        if x_pos + current_token_width_estimate > fig_width_norm:
             x_pos = 0.01 # Reset to left edge + padding
             y_pos -= V_PADDING # Move to the next line
             max_y_reached = min(max_y_reached, y_pos) # Track lowest point reached


        # Plot the token text
        ax.text(x_pos, y_pos, token_text,
                color=text_color,
                fontsize=FONT_SIZE,
                family='monospace', # Use monospace font for better width consistency
                ha='left',
                va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc=bg_color, ec='grey', lw=0.5, alpha=0.8))

        # Update position for the next token
        x_pos += current_token_width_estimate

        # Safety break
        if y_pos < 0.05: # Stop if getting too close to bottom
             logging.warning("Plotting truncated: Exceeded maximum lines.")
             ax.text(x_pos, y_pos, "...", color='red', fontsize=FONT_SIZE, va='top', ha='left')
             break

    # Return Y coordinate below the lowest point reached by this block
    return max_y_reached - V_PADDING


# --- Main Processing Loop ---
num_samples = delta_w.shape[0] if isinstance(delta_w, np.ndarray) and delta_w.ndim >= 1 else 0
if num_samples == 0: logging.info("No samples found."); exit()

logging.info(f"Processing {num_samples} samples using Matplotlib (with text decoding)...")
processed_count = 0; error_count = 0

for i in range(num_samples): # Use full range: range(num_samples)
    logging.info(f"--- Processing Sample {i} ---")
    try:
        # --- Get Data ---
        current_delta_w = delta_w[i, 0, :] if delta_w.ndim == 3 and delta_w.shape[1] > 0 else np.array([])
        raw_r = right_indices[i] if i < len(right_indices) else []
        raw_w = wrong_indices[i] if i < len(wrong_indices) else []
        r_ids = [int(t) for t in raw_r if t is not None and isinstance(t, (int, float, np.number, str)) and str(t).isdigit()]
        w_ids = [int(t) for t in raw_w if t is not None and isinstance(t, (int, float, np.number, str)) and str(t).isdigit()]

        max_vocab_idx = current_delta_w.shape[0] - 1 if current_delta_w.size > 0 else -1
        valid_r_ids = [tid for tid in r_ids if 0 <= tid <= max_vocab_idx]
        logging.debug(f"Sample {i}: Wrong IDs: {len(w_ids)}, Valid Right IDs: {len(valid_r_ids)} (of {len(r_ids)})")

        # --- Colormap Setup ---
        cmap = matplotlib.cm.get_cmap('coolwarm')
        norm_right = None; delta_values = []
        if valid_r_ids and max_vocab_idx >= 0:
            try:
                delta_values = [float(current_delta_w[tid]) for tid in valid_r_ids]
                if delta_values:
                    min_dw, max_dw = min(delta_values), max(delta_values)
                    logging.debug(f"Sample {i}: Right delta_w range: [{min_dw:.4f}, {max_dw:.4f}]")
                    if abs(min_dw - max_dw) > 1e-9:
                        norm_right = matplotlib.colors.Normalize(vmin=min_dw, vmax=max_dw)
            except (IndexError, ValueError) as e:
                logging.warning(f"Sample {i}: Error getting delta_w values: {e}. Right tokens won't be colored.")
                delta_values = []

        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        fig.suptitle(f"Sample {i} Token Visualization (Decoded)", fontsize=14)
        current_y = 1.0 - TITLE_SPACE

        # --- Plot Wrong Tokens ---
        ax.text(0.01, current_y + 0.01, "Wrong Tokens:", fontsize=FONT_SIZE+1, weight='bold', va='bottom', ha='left')
        wrong_color_info = {'type': 'fixed', 'color': '#333333', 'bgcolor': '#f0f0f0'}
        current_y = plot_tokens_matplotlib_text(ax, w_ids, current_y, wrong_color_info, 1.0, tokenizer) # Pass tokenizer

        # --- Plot Right Tokens ---
        current_y -= V_PADDING # Extra space
        ax.text(0.01, current_y + 0.01, "Right Tokens (Colored by delta_w):", fontsize=FONT_SIZE+1, weight='bold', va='bottom', ha='left')
        right_color_info = {'type': 'dynamic', 'cmap': cmap, 'norm': norm_right, 'values': delta_values, 'bgcolor': '#e8f6f3'}
        current_y = plot_tokens_matplotlib_text(ax, valid_r_ids, current_y, right_color_info, 1.0, tokenizer) # Pass tokenizer

        # --- Add Color Legend ---
        if delta_values and norm_right:
            try:
                cax = fig.add_axes([0.15, 0.06, 0.7, 0.02]) # Adjusted position/size
                cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_right, cmap=cmap), cax=cax, orientation='horizontal')
                cb.set_label('delta_w value', fontsize=FONT_SIZE)
                cb.ax.tick_params(labelsize=FONT_SIZE-1)
                # fig.subplots_adjust(bottom=0.12) # Adjust plot area if needed
            except Exception as e:
                logging.warning(f"Sample {i}: Could not draw colorbar. Error: {e}")
        elif delta_values: # Single value case
             ax.text(0.5, 0.02, f"Right Tokens delta_w (all): {delta_values[0]:.3f}",
                     ha='center', va='bottom', transform=fig.transFigure, fontsize=FONT_SIZE-1, color='grey')


        # --- Save PNG ---
        png_filename = os.path.join(output_dir_png, f"sample_{i}_vis_matplotlib_text.png")
        plt.savefig(png_filename, dpi=150, bbox_inches='tight')
        logging.info(f"Saved PNG: {png_filename}")
        processed_count += 1

    except Exception as e:
        logging.error(f"Sample {i}: Failed to process or plot. Error: {e}", exc_info=True)
        error_count += 1
    finally:
        plt.close(fig) # Ensure figure is closed even on error


logging.info("-" * 30)
logging.info(f"Finished.")
logging.info(f"Successfully generated {processed_count} PNG files.")
if error_count > 0: logging.warning(f"Encountered errors processing {error_count} samples.")
logging.info(f"Output saved in: {output_dir_png}")
logging.info("-" * 30)
