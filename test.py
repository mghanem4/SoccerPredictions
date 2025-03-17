import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def write_text_to_pdf(pdf: PdfPages, text_lines: str, lines_per_page=40, title=False, header=False, fontsize=10, lines_on_page=0, num_pages=0):
    text_lines = text_lines.split('\n')

    if pdf is not None:
        # Check if we need to create a new figure
        if lines_on_page == 0:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
        else:
            fig = plt.gcf()
            ax = plt.gca()

        for line in text_lines:
            # Handle title (always on a new page)
            if title:
                # Save the current page if there are lines on it
                if lines_on_page > 0 or num_pages > 0:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    num_pages += 1
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')

                # Write the title centered on a new page
                ax.text(0.5, 0.5, line, fontsize=16, ha='center', va='center')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                num_pages += 1

                # Prepare for a new page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                lines_on_page = 0
                continue

            # Check if a new page is needed for regular or header text
            if lines_on_page >= lines_per_page:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                num_pages += 1
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                lines_on_page = 0

            # Handle header text formatting (larger font)
            if header:
                ax.text(0.5, 1 - (lines_on_page + 1) * 0.025, line, fontsize=14, va='top', ha='center')
            else:
                ax.text(0.1, 1 - (lines_on_page + 1) * 0.025, line, fontsize=fontsize, va='top', ha='left')

            lines_on_page += 1

    return [lines_on_page, num_pages]


def increment_counters(arr1: list, arr2: list) -> list:
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr1[i] += arr2[i]
    else:
        print("Error: Arrays must be of the same length")
        exit(0)
    return arr1


# Create a PdfPages object to write to a PDF file
pdf_path = 'output_test.pdf'
with PdfPages(pdf_path) as pdf:
    totals = [0, 0]  # [lines_on_page, num_pages]

    vars = write_text_to_pdf(pdf, "This is a Title", title=True, lines_on_page=totals[0], num_pages=totals[1])
    totals = increment_counters(totals, vars)

    vars = write_text_to_pdf(pdf, "This is a Header", header=True, lines_on_page=totals[0], num_pages=totals[1])
    totals = increment_counters(totals, vars)

    vars = write_text_to_pdf(pdf, "This is a test line 1.\nThis is a test line 2.\nThis is a test line 3.", lines_on_page=totals[0], num_pages=totals[1])
    totals = increment_counters(totals, vars)

    # Save the last page
    if totals[0] > 0:
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()

print("Test PDF created successfully at:", pdf_path)
