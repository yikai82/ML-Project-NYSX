## >> save as html and png for sharing
FILE_NAME = f"pca_{X}_vs_{Y}_test10"
fig.write_html(f"{HTML_PATH}/{FILE_NAME}.html")
fig.write_image(f"{FIG_PATH}/{FILE_NAME}.png",
               width=XXX,
               height=XXX,
               scale=3          # Scale factor (like DPI multiplier)
               )

# allow to open automatically in the default browser
import webbrowser
webbrowser.open(f"{HTML_PATH}/{FILE_NAME}.html")
