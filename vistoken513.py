import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from transformers import AutoTokenizer

# --- Basic Configuration ---
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(filename)s:%(lineno)d: %(message)s') # DEBUG to see plot details
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # INFO for less verbose output
os.environ['HF_HOME'] = '/ssdwork/huyang/r1'

# --- Set Font ---
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    logging.info(f"Set Matplotlib font to: {plt.rcParams['font.family']}")
except Exception as e:
    logging.warning(f"Could not set default font: {e}")

# --- Define File Paths ---
base_dir = "/ssdwork/huyang/r1/simple_GRPO_debug/simple_grpo_v1_index_compare/" # ADJUST
right_indices_path = os.path.join(base_dir, "right.npy")
wrong_indices_path = os.path.join(base_dir, "wrong.npy")
delta_w_path = os.path.join(base_dir, "delta_w.npy")
tokenizer_path = "/ssdwork/huyang/r1/simple_GRPO_debug/simple_grpo_v1/step_100" # ADJUST

# --- Output Directory ---
output_dir_png = os.path.join(base_dir, "visualizations_png_matplotlib_text_adjusted")

# --- Matplotlib Plotting Constants ---
FIG_WIDTH_INCHES = 18
FIG_HEIGHT_INCHES = 10
FONT_SIZE = 8
H_CHAR_FACTOR = 0.65 # Character width factor
V_PADDING = 0.030 # Vertical padding
TITLE_SPACE = 0.05
NEUTRAL_COLOR = '#888888'
BBOX_PAD = 0.15 # Bbox padding (as a fraction of fontsize, used in points) - VALUE used in f-string now
INTER_TOKEN_H_PAD_NORM = 0.004 # Horizontal padding

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
    logging.warning(f"Failed to load tokenizer: {e}", exc_info=True)
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
    # ... (Keep implementation from previous step) ...
    if not tokenizer: return f"ID:{token_id}"
    try:
        token_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        token_str = token_str.replace(' ', '␣').replace('\n', '↵').replace('\t', '→')
        if not token_str:
             vocab = getattr(tokenizer, 'vocab', None)
             if vocab: inv_vocab = {v: k for k, v in vocab.items()}; raw_token = inv_vocab.get(token_id, f"[UNK:{token_id}]"); return f"[{raw_token.strip()}]"
             else: return f"[EMPTY:{token_id}]"
        safe_chars = ['␣', '↵', '→']
        cleaned_token_str = ''.join(c if c.isprintable() or c in safe_chars else '□' for c in token_str)
        return cleaned_token_str
    except Exception as e: logging.warning(f"Decode err ID {token_id}: {e}", exc_info=False); return f"[ERR:{token_id}]"

# --- Helper Function to Plot Tokens (Corrected Bbox) ---
def plot_tokens_matplotlib_text(fig, ax, tokens, start_y, color_info, fig_width_norm, tokenizer):
    if not tokens:
         ax.text(0.01, start_y, "(None)", fontsize=FONT_SIZE, color='grey', va='top', ha='left')
         return start_y - V_PADDING

    char_width_estimate_norm = (FONT_SIZE / 72) * H_CHAR_FACTOR / FIG_WIDTH_INCHES
    bbox_pad_points = BBOX_PAD * FONT_SIZE
    bbox_pad_norm = (bbox_pad_points / 72) / FIG_WIDTH_INCHES * 2

    x_pos, y_pos = 0.01, start_y
    max_y_reached = start_y

    for idx, token_id in enumerate(tokens):
        token_text = decode_token_safe(tokenizer, token_id)

        text_color = 'black'; bg_color = 'white'
        try: # Determine color logic (same as before)
            if color_info['type'] == 'fixed': text_color, bg_color = color_info.get('color', 'black'), color_info.get('bgcolor', 'white')
            elif color_info['type'] == 'dynamic':
                bg_color = color_info.get('bgcolor', 'white')
                values = color_info.get('values', []); norm = color_info.get('norm'); cmap = color_info.get('cmap')
                if idx < len(values):
                    value = values[idx]; is_zero = abs(value) < 1e-9
                    if is_zero: text_color = NEUTRAL_COLOR
                    elif norm and cmap: text_color = cmap(norm(value))
                    elif values and cmap: text_color = NEUTRAL_COLOR
        except Exception as e: logging.warning(f"Color err ID {token_id}: {e}"); text_color = 'red'

        current_token_width_estimate = (len(token_text) * char_width_estimate_norm) + bbox_pad_norm
        if x_pos + current_token_width_estimate + INTER_TOKEN_H_PAD_NORM > fig_width_norm:
            x_pos = 0.01; y_pos -= V_PADDING
            max_y_reached = min(max_y_reached, y_pos)

        logging.debug(f"Plot token ID {token_id}: Text='{token_text}', X={x_pos:.3f}, Y={y_pos:.3f}, EstWidth={current_token_width_estimate:.3f}")

        # Plot the token text with CORRECTED bbox
        ax.text(x_pos, y_pos, token_text,
                color=text_color,
                fontsize=FONT_SIZE,
                family=plt.rcParams['font.family'],
                ha='left', va='top',
                # *** CORRECTED THIS LINE ***
                bbox=dict(boxstyle=f'round,pad={BBOX_PAD}', fc=bg_color, ec='grey', lw=0.5, alpha=0.8))

        x_pos += current_token_width_estimate + INTER_TOKEN_H_PAD_NORM
        if y_pos < 0.1: logging.warning("Plot truncated"); ax.text(x_pos, y_pos, "...", color='red'); break

    return max_y_reached - (V_PADDING / 2)


# --- Main Processing Loop ---
num_samples = delta_w.shape[0] if isinstance(delta_w, np.ndarray) and delta_w.ndim >= 1 else 0
if num_samples == 0: logging.info("No samples found."); exit()

logging.info(f"Processing {num_samples} samples...")
processed_count = 0; error_count = 0

for i in range(num_samples):
    logging.info(f"--- Processing Sample {i} ---")
    try:
        # --- Get Data ---
        current_delta_w = delta_w[i, 0, :] if delta_w.ndim == 3 and delta_w.shape[1] > 0 else np.array([])
        raw_r = right_indices[i] if i < len(right_indices) else []; raw_w = wrong_indices[i] if i < len(wrong_indices) else []
        r_ids = [int(t) for t in raw_r if t is not None and isinstance(t, (int, float, np.number, str)) and str(t).isdigit()]
        w_ids = [int(t) for t in raw_w if t is not None and isinstance(t, (int, float, np.number, str)) and str(t).isdigit()]
        max_vocab_idx = current_delta_w.shape[0] - 1 if current_delta_w.size > 0 else -1
        valid_r_ids = [tid for tid in r_ids if 0 <= tid <= max_vocab_idx]

        # --- Colormap Setup (Centered at Zero) ---
        cmap = cm.get_cmap('coolwarm')
        norm_right = None; delta_values = []
        if valid_r_ids and max_vocab_idx >= 0:
            try: # NaN Filtering and Normalization (same as before)
                valid_delta_indices = [idx for idx, tid in enumerate(valid_r_ids) if not np.isnan(current_delta_w[tid])]
                valid_r_ids_nonan = [valid_r_ids[idx] for idx in valid_delta_indices]
                delta_values = [float(current_delta_w[valid_r_ids_nonan[idx]]) for idx in range(len(valid_r_ids_nonan))]
                if valid_r_ids_nonan != valid_r_ids: logging.warning(f"Sample {i}: Excluded {len(valid_r_ids) - len(valid_r_ids_nonan)} NaN delta_w."); valid_r_ids = valid_r_ids_nonan
                if delta_values:
                    min_dw, max_dw = min(delta_values), max(delta_values); vabs = max(abs(min_dw), abs(max_dw))
                    if vabs > 1e-9: norm_right = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs); logging.debug(f"Sample {i}: Centered norm: vabs={vabs:.4f}")
                    else: logging.debug(f"Sample {i}: All delta_w near zero.")
                else: logging.debug(f"Sample {i}: No valid delta values.")
            except (IndexError, ValueError) as e: logging.warning(f"Sample {i}: Err delta_w: {e}"); delta_values = []; valid_r_ids = []

        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        current_y = 1.0 # Start from top

        # --- Add Horizontal Colorbar at the TOP (if needed) ---
        colorbar_height_norm = 0.02 # Height of the horizontal colorbar
        colorbar_v_pad = 0.008 # REDUCED Padding below the colorbar
        if norm_right:
            try:
                cax = fig.add_axes([0.15, current_y - colorbar_height_norm - 0.01, 0.7, colorbar_height_norm]) # Position near top
                mappable = cm.ScalarMappable(norm=norm_right, cmap=cmap)
                cb = plt.colorbar(mappable, cax=cax, orientation='horizontal')
                cb.ax.tick_params(labelsize=FONT_SIZE-1)
                current_y -= (colorbar_height_norm + colorbar_v_pad) # Move starting Y down
                logging.debug(f"Sample {i}: Placed horizontal colorbar at top, new current_y={current_y:.3f}")
            except Exception as e:
                logging.warning(f"Sample {i}: Could not draw colorbar at top: {e}")
                current_y -= (colorbar_height_norm + colorbar_v_pad) # Still adjust Y even if drawing failed, for consistent layout
        else:
             current_y -= colorbar_v_pad # ADJUSTED padding if no colorbar

        # --- Plot Wrong Tokens ---
        ax.text(0.01, current_y, "Wrong:", fontsize=FONT_SIZE+1, weight='bold', va='bottom', ha='left') # Use current_y
        wrong_color_info = {'type': 'fixed', 'color': '#333333', 'bgcolor': '#f0f0f0'}
        # Use current_y - small offset for plotting start, plot area width back to 1.0
        y_after_wrong = plot_tokens_matplotlib_text(fig, ax, w_ids, current_y - 0.01, wrong_color_info, 1.0, tokenizer)
        logging.debug(f"Sample {i}: Y after wrong tokens: {y_after_wrong:.3f}")
        current_y = y_after_wrong # Update current_y to the bottom of the wrong section

        # --- Plot Right Tokens ---
        current_y -= V_PADDING * 1.5 # INCREASED vertical spacing
        ax.text(0.01, current_y, "Right:", fontsize=FONT_SIZE+1, weight='bold', va='bottom', ha='left') # Use current_y
        right_color_info = {'type': 'dynamic', 'cmap': cmap, 'norm': norm_right, 'values': delta_values, 'bgcolor': '#e8f6f3'}
        # Use current_y - small offset for plotting start, plot area width back to 1.0
        y_pos_after_right = plot_tokens_matplotlib_text(fig, ax, valid_r_ids, current_y - 0.01, right_color_info, 1.0, tokenizer)
        logging.debug(f"Sample {i}: Y after right tokens: {y_pos_after_right:.3f}")

        # --- Save PNG and PDF ---
        base_filename = os.path.join(output_dir_png, f"sample_{i}_vis_matplotlib_text_adjusted")
        png_filename = f"{base_filename}.png"
        pdf_filename = f"{base_filename}.pdf"

        try:
            plt.savefig(png_filename, dpi=180, bbox_inches='tight')
            logging.info(f"Saved PNG: {png_filename}")
            plt.savefig(pdf_filename, bbox_inches='tight') # Save PDF
            logging.info(f"Saved PDF: {pdf_filename}")
            processed_count += 1
        except Exception as e:
             logging.error(f"Sample {i}: Failed to save files: {e}", exc_info=True)
             error_count += 1 # Count saving errors as processing errors

    except Exception as e:
        logging.error(f"Sample {i}: Failed plot/process: {e}", exc_info=True)
        error_count += 1
    finally:
        plt.close(fig) # Ensure figure is closed

# --- Final Summary ---
logging.info("-" * 30); logging.info("Finished.")
logging.info(f"Successfully generated {processed_count} PNG files.")
if error_count > 0: logging.warning(f"Errors processing {error_count} samples.")
logging.info(f"Output saved in: {output_dir_png}"); logging.info("-" * 30)
