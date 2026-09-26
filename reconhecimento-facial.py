# ============================================================
# SISTEMA DE PORTARIA COM RECONHECIMENTO FACIAL
# ============================================================

import cv2
from deepface import DeepFace
import time
import os
import pygame


# ============================================================
# 1. CONFIGURAR ÁUDIOS
# ============================================================

# Inicializa o sistema de áudio
pygame.mixer.init()

audio_liberado = os.path.join(
    "audio",
    "WhatsApp-Audio-2026-09-26-at-12.14.57.mp3"
)

audio_negado = os.path.join(
    "audio",
    "WhatsApp-Audio-2026-09-26-at-12.15.13.mp3"
)


# Verifica se os áudios existem
if not os.path.exists(audio_liberado):
    print(f"Erro: áudio não encontrado: {audio_liberado}")
    exit()

if not os.path.exists(audio_negado):
    print(f"Erro: áudio não encontrado: {audio_negado}")
    exit()


# ============================================================
# FUNÇÕES PARA TOCAR OS ÁUDIOS
# ============================================================

def tocar_audio(caminho):
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(caminho)
        pygame.mixer.music.play()

    except Exception as e:
        print(f"Erro ao reproduzir áudio: {e}")


# ============================================================
# 2. CARREGAR A IDENTIDADE DO MORADOR
# ============================================================

imagem_referencia = "face_id.jpeg"

print("Carregando identidade do morador...")


# Verifica se a imagem de referência existe
if not os.path.exists(imagem_referencia):

    print(f"Erro: arquivo '{imagem_referencia}' não encontrado.")
    exit()


try:

    # Faz uma pré-análise da imagem de referência
    # para verificar se existe um rosto nela.
    DeepFace.represent(
        img_path=imagem_referencia,
        model_name="VGG-Face",
        enforce_detection=True
    )

    print("Identidade carregada com sucesso!")


except Exception as e:

    print(f"Erro ao carregar a identidade: {e}")
    exit()


# ============================================================
# 3. INICIAR A CÂMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Erro: não foi possível abrir a câmera.")
    exit()


print("Sistema de Portaria Ativo.")
print("Aproxime-se da câmera para realizar o reconhecimento.")
print("Pressione 'q' para sair.")


# ============================================================
# 4. CONFIGURAÇÕES
# ============================================================

# Tempo entre verificações
tempo_ultima_verificacao = 0

# Verificar a cada 5 segundos
intervalo_verificacao = 5


# Estado do acesso
acesso_liberado = False
acesso_negado = False


# Tempo em que o resultado foi mostrado
tempo_resultado = 0


# Quanto tempo o resultado permanece na tela
duracao_resultado = 2


# ============================================================
# 5. LOOP PRINCIPAL DA CÂMERA
# ============================================================

while True:

    # Captura imagem da câmera
    ret, frame_original = cap.read()


    if not ret:

        print("Erro ao capturar imagem da câmera.")
        break


    # ========================================================
    # ESPELHAR A IMAGEM
    # ========================================================

    frame = cv2.flip(frame_original, 1)


    # Dimensões da imagem
    height, width, _ = frame.shape


    # ========================================================
    # ÁREA DE LEITURA
    # ========================================================

    cv2.rectangle(
        frame,
        (100, 100),
        (width - 100, height - 100),
        (255, 0, 0),
        2
    )


    # ========================================================
    # 6. VERIFICAÇÃO AUTOMÁTICA
    # ========================================================

    tempo_atual = time.time()


    # Verifica se já passaram 5 segundos
    if tempo_atual - tempo_ultima_verificacao >= intervalo_verificacao:

        tempo_ultima_verificacao = tempo_atual


        try:

            print("Procurando rosto...")


            # ------------------------------------------------
            # DETECTAR ROSTO
            # ------------------------------------------------

            rostos = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )


            rosto_detectado = False


            # Verifica se realmente encontrou um rosto
            for rosto in rostos:

                confidence = rosto.get("confidence", 0)


                if confidence > 0.5:

                    rosto_detectado = True
                    break


            # ------------------------------------------------
            # SE ENCONTROU ROSTO, FAZER RECONHECIMENTO
            # ------------------------------------------------

            if rosto_detectado:

                print("Rosto detectado!")
                print("Verificando identidade...")


                resultado = DeepFace.verify(

                    img1_path=frame,

                    img2_path=imagem_referencia,

                    model_name="VGG-Face",

                    enforce_detection=False
                )


                # ====================================================
                # PESSOA RECONHECIDA
                # ====================================================

                if resultado["verified"]:

                    print(">>> ACESSO LIBERADO!")


                    acesso_liberado = True
                    acesso_negado = False

                    tempo_resultado = time.time()


                    # 🔊 TOCAR ÁUDIO DE ACESSO LIBERADO
                    tocar_audio(audio_liberado)


                # ====================================================
                # PESSOA NÃO RECONHECIDA
                # ====================================================

                else:

                    print(">>> ACESSO NEGADO!")


                    acesso_liberado = False
                    acesso_negado = True

                    tempo_resultado = time.time()


                    # 🔊 TOCAR ÁUDIO DE ACESSO NEGADO
                    tocar_audio(audio_negado)


        except Exception as e:

            print(f"Erro na leitura: {e}")


    # ============================================================
    # 7. MOSTRAR ACESSO LIBERADO
    # ============================================================

    if acesso_liberado:

        # Borda verde
        cv2.rectangle(

            frame,

            (0, 0),

            (width, height),

            (0, 255, 0),

            10
        )


        # Texto
        cv2.putText(

            frame,

            "ACESSO LIBERADO",

            (width // 2 - 220, height // 2),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.5,

            (0, 255, 0),

            4
        )


    # ============================================================
    # 8. MOSTRAR ACESSO NEGADO
    # ============================================================

    elif acesso_negado:

        # Borda vermelha
        cv2.rectangle(

            frame,

            (0, 0),

            (width, height),

            (0, 0, 255),

            10
        )


        # Texto
        cv2.putText(

            frame,

            "ACESSO NEGADO",

            (width // 2 - 200, height // 2),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.5,

            (0, 0, 255),

            4
        )


    # ============================================================
    # 9. REMOVER RESULTADO APÓS ALGUNS SEGUNDOS
    # ============================================================

    if acesso_liberado or acesso_negado:

        if time.time() - tempo_resultado > duracao_resultado:

            acesso_liberado = False
            acesso_negado = False


    # ============================================================
    # 10. MENSAGEM NORMAL
    # ============================================================

    if not acesso_liberado and not acesso_negado:

        cv2.putText(

            frame,

            "Aproxime-se da camera",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )


    # ============================================================
    # 11. MOSTRAR IMAGEM
    # ============================================================

    cv2.imshow(

        "Portaria",

        frame
    )


    # ============================================================
    # 12. TECLA PARA SAIR
    # ============================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# 13. ENCERRAR
# ============================================================

cap.release()

cv2.destroyAllWindows()

pygame.mixer.quit()

print("Sistema encerrado.")