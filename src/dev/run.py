
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import load_model
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from scipy import signal


num_rows = 99
num_columns = 174
num_channels = 1
max_pad_len = 174

mapping_list = ['', 'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',  'jackhammer', 'siren', 'street_music']
mapping_list_4_classes = ['car_horn', 'dog_bark', 'gun_shot',  'siren']
model_v2 = load_model('cnn_v2.h5')
model_4_classes = load_model('cnn_4_classes_v2.h5')

def extract_features(file_name):
   
    try:
        sample_rate,audio=wav.read(file_name)

        audio = audio.astype(np.float32, order='C') / 32768.00
        
        # transform to monochannel
        try:
            d = (audio[:,0] + audio[:,1]) / 2
            f = signal.resample(d, 22050)
        except:
            f = signal.resample(audio, 22050)

        mfccs = mfcc(f, samplerate =22050, numcep=20,nfilt=26,nfft=1024, appendEnergy=False)

        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None
     

def get_prediction(file_name, map_list, model):
    
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict(prediction_feature)
    classes_x=np.argmax(predicted_vector, axis=1)
    indice = classes_x[0]
    
    max_value = np.amax(predicted_vector)
    
    print('max value ', max_value)
    
    
    
    pred_class = map_list[indice]
    

    if max_value < 0.70:
        print("Sem certeza")
        print(f"The predicted class possible is: {pred_class}") 
        return ''
    
    print(f"The predicted class is: {pred_class}") 
    
    print(predicted_vector)
    pred_vector = predicted_vector.tolist()
    print(dict(zip(map_list, pred_vector[0])))    
    
    return pred_class

# Esse é mais simples. mas parece que o pyaudio funciona melhor para rasp? Checar
def save_sdaudio_file(samplerate, duration, filename):

    print('* init rec')
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=1, blocking=True)
    print('* end rec')
    sf.write(filename, mydata, samplerate)


#TODO ao inves de pegar o  maior so pega  se %>x

_modo = 1
temp_filename = 'temp_output.wav'

# Pode ser atualizado depois que enviarem do app ou algo assim, deixei auqiu para mostrar so
def set_modo(modo=None):
    global _modo
    if modo is not None:
        _modo = modo

def run_once(modo):
    
    set_modo(modo)
    
    if _modo == 0:
        exit(0) # finaliza, da para colocar para iniciar / desligar no app
        
    if _modo == 1: # faz cada X segundos
        save_sdaudio_file(16000, 4, temp_filename)
        pred = get_prediction(temp_filename, mapping_list, model_v2) # Pode ser alterado para pegar a variavel so
        return pred
        
    if _modo == 2: # Split por silencio
        save_sdaudio_file(16000, 4, temp_filename)
        pred = get_prediction(temp_filename, mapping_list_4_classes, model_4_classes) # Pode ser alterado para pegar a variavel so
        return pred
    
ligado = True # Pode ser settado pelo app ou algo assim

while ligado:
    pred = run_once(2)
    
    # pred ta ai
    print(pred)

