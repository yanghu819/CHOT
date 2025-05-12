import re
import ast
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def parse_log_file(log_file_path):
    all_logit_changes = []
    current_sample_id = None

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample_match = re.search(r"--- Sample (\d+) ---", line)
            if sample_match:
                current_sample_id = sample_match.group(1)
                continue

            top_tokens_match = re.search(r"Logits Change Top 5 Tokens \(token, value\): (\[.*\])", line)
            if top_tokens_match and current_sample_id:
                try:
                    tokens_str = top_tokens_match.group(1)
                    tokens_list = ast.literal_eval(tokens_str)
                    for token, value in tokens_list:
                        all_logit_changes.append({'sample_id': current_sample_id, 'token': token, 'value': float(value), 'type': 'top'})
                except Exception as e:
                    print(f"Error parsing top tokens for sample {current_sample_id}: {line.strip()} -> {e}")

            bottom_tokens_match = re.search(r"Logits Change Bottom 5 Tokens \(token, value - smallest first\): (\[.*\])", line)
            if bottom_tokens_match and current_sample_id:
                try:
                    tokens_str = bottom_tokens_match.group(1)
                    tokens_list = ast.literal_eval(tokens_str)
                    for token, value in tokens_list:
                        all_logit_changes.append({'sample_id': current_sample_id, 'token': token, 'value': float(value), 'type': 'bottom'})
                except Exception as e:
                    print(f"Error parsing bottom tokens for sample {current_sample_id}: {line.strip()} -> {e}")
    
    return all_logit_changes

def analyze_changes(logit_changes):
    if not logit_changes:
        print("No logit changes found.")
        return

    # font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    font_path = './wqy-zenhei.ttc'
    font_prop = None
    try:
        font_prop = fm.FontProperties(fname=font_path)
        print(f"Successfully loaded FontProperties from: {font_path} (Font name: {font_prop.get_name()})")
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"CRITICAL: Could not load FontProperties from {font_path} - {e}. Chinese characters will likely not render.")
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        print("Using fallback generic font list for sans-serif due to error with specific font file.")

    # Data preparation for frequent tokens
    sample_top_tokens_freq = defaultdict(int)
    sample_top_token_sum_value = defaultdict(float)
    sample_top_token_count = defaultdict(int)

    sample_bottom_tokens_freq = defaultdict(int)
    sample_bottom_token_sum_value = defaultdict(float)
    sample_bottom_token_count = defaultdict(int)

    for change in logit_changes:
        if change['type'] == 'top':
            sample_top_tokens_freq[change['token']] += 1
            sample_top_token_sum_value[change['token']] += change['value']
            sample_top_token_count[change['token']] += 1
        elif change['type'] == 'bottom':
            sample_bottom_tokens_freq[change['token']] += 1
            sample_bottom_token_sum_value[change['token']] += change['value']
            sample_bottom_token_count[change['token']] += 1

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(30, 16))
    
    fig_suptitle_kwargs = {'fontsize': 30, 'y': 0.98}
    if font_prop: fig_suptitle_kwargs['fontproperties'] = font_prop
    # fig.suptitle('Logit Token Frequency Analysis', **fig_suptitle_kwargs)

    ax = axs[0]
    # title_kwargs = {'fontsize': 26} # No longer needed for individual subplot titles
    label_kwargs = {'fontsize': 24}
    tick_label_kwargs = {'fontsize': 26}
    xtick_label_kwargs = {'fontsize': 22}
    text_fallback_kwargs = {'fontsize': 24}

    if font_prop:
        # title_kwargs['fontproperties'] = font_prop # No longer needed
        label_kwargs['fontproperties'] = font_prop
        text_fallback_kwargs['fontproperties'] = font_prop

    if sample_top_tokens_freq:
        sorted_top_freq_items = sorted(sample_top_tokens_freq.items(), key=lambda item: item[1], reverse=True)[:10]
        top_freq_tokens = [item[0] for item in sorted_top_freq_items]
        top_freq_counts = [item[1] for item in sorted_top_freq_items]
        ax.barh(top_freq_tokens, top_freq_counts, color='mediumseagreen')
        # ax.set_title('Most Frequent Tokens in Sample Top Lists (Top 10)', **title_kwargs) # Removed subplot title
        ax.invert_yaxis()
        ax.set_xlabel('Frequency in Sample Top Lists', **label_kwargs)
        if font_prop:
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
                label.set_fontsize(tick_label_kwargs['fontsize'])
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_prop)
                label.set_fontsize(xtick_label_kwargs['fontsize'])
        else:
            ax.tick_params(axis='y', labelsize=tick_label_kwargs['fontsize'])
            ax.tick_params(axis='x', labelsize=xtick_label_kwargs['fontsize'])
    else:
        ax.text(0.5, 0.5, "No 'Top 5' token data found", ha='center', va='center', **text_fallback_kwargs)
        # ax.set_title('Most Frequent Tokens in Sample Top Lists (Top 10)', **title_kwargs) # Removed subplot title

    ax = axs[1]
    if sample_bottom_tokens_freq:
        sorted_bottom_freq_items = sorted(sample_bottom_tokens_freq.items(), key=lambda item: item[1], reverse=True)[:10]
        bottom_freq_tokens = [item[0] for item in sorted_bottom_freq_items]
        bottom_freq_counts = [item[1] for item in sorted_bottom_freq_items]
        ax.barh(bottom_freq_tokens, bottom_freq_counts, color='goldenrod')
        # ax.set_title('Most Frequent Tokens in Sample Bottom Lists (Top 10)', **title_kwargs) # Removed subplot title
        ax.invert_yaxis()
        ax.set_xlabel('Frequency in Sample Bottom Lists', **label_kwargs)
        if font_prop:
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
                label.set_fontsize(tick_label_kwargs['fontsize'])
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_prop)
                label.set_fontsize(xtick_label_kwargs['fontsize'])
        else:
            ax.tick_params(axis='y', labelsize=tick_label_kwargs['fontsize'])
            ax.tick_params(axis='x', labelsize=xtick_label_kwargs['fontsize'])
    else:
        ax.text(0.5, 0.5, "No 'Bottom 5' token data found", ha='center', va='center', **text_fallback_kwargs)
        # ax.set_title('Most Frequent Tokens in Sample Bottom Lists (Top 10)', **title_kwargs) # Removed subplot title

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Save plots
    output_base = "logitchange_freq"
    png_file = f"{output_base}.png"
    pdf_file = f"{output_base}.pdf"

    try:
        plt.savefig(png_file)
        plt.savefig(pdf_file)
        print(f"Plots saved to {png_file} and {pdf_file} in the script's directory.")
    except Exception as e:
        print(f"Error saving plots: {e}")
    finally:
        plt.close(fig)
    
    # Console output for frequent tokens
    if sample_top_tokens_freq:
        print(f"\n--- Tokens Most Frequently in any Sample's 'Top 5 Logits Change' List (Top 10) ---")
        sorted_top_freq = sorted(sample_top_tokens_freq.items(), key=lambda item: item[1], reverse=True)
        for token, count in sorted_top_freq[:10]:
            avg_val = sample_top_token_sum_value[token] / sample_top_token_count[token]
            print(f"Token: '{token}', Frequency: {count}, Avg Value when in Top: {avg_val:.4f}")
    
    if sample_bottom_tokens_freq:
        print(f"\n--- Tokens Most Frequently in any Sample's 'Bottom 5 Logits Change' List (Top 10) ---")
        sorted_bottom_freq = sorted(sample_bottom_tokens_freq.items(), key=lambda item: item[1], reverse=True)
        for token, count in sorted_bottom_freq[:10]:
            avg_val = sample_bottom_token_sum_value[token] / sample_bottom_token_count[token]
            print(f"Token: '{token}', Frequency: {count}, Avg Value when in Bottom: {avg_val:.4f}")

if __name__ == "__main__":
    log_file = "eval_only_slot_5_0.log"
    logit_data = parse_log_file(log_file)
    if logit_data:
        analyze_changes(logit_data)
    else:
        print(f"No data parsed from {log_file}") 
