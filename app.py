import gradio as gr


def process_data(nom, prenom, audio):
    # Traitez les données ici
    # audio est le chemin du fichier audio téléchargé
    message = f"Bonjour {prenom} {nom} !"

    # Vous pouvez ajouter un retour avec l'audio ou d'autres traitements
    return message, audio


# Création de l'interface
with gr.Blocks(title="Formulaire Audio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Formulaire avec Audio")
    gr.Markdown("Veuillez remplir vos informations et télécharger un fichier audio.")

    with gr.Row():
        with gr.Column():
            # Champs de saisie
            nom_input = gr.Textbox(
                label="Nom",
                placeholder="Entrez votre nom",
                scale=1
            )

            prenom_input = gr.Textbox(
                label="Prénom",
                placeholder="Entrez votre prénom",
                scale=1
            )

            # Input audio
            audio_input = gr.Audio(
                label="Téléchargez un fichier audio",
                type="filepath",
                sources=["upload", "microphone"]
            )

            # Bouton de soumission
            submit_btn = gr.Button("Soumettre", variant="primary")

        with gr.Column():
            # Sorties
            output_text = gr.Textbox(label="Message", interactive=False)
            output_audio = gr.Audio(label="Audio reçu")

    # Lier le bouton à la fonction
    submit_btn.click(
        fn=process_data,
        inputs=[nom_input, prenom_input, audio_input],
        outputs=[output_text, output_audio]
    )

    # Exemple
    gr.Examples(
        examples=[
            ["Dupont", "Jean", None],
            ["Martin", "Sophie", None],
        ],
        inputs=[nom_input, prenom_input, audio_input]
    )

# Lancer l'application
demo.launch()