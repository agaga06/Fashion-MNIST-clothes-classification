from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential #jest tez funkcjonalny model budowania modelu
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import mnist_reader


def draw(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], "r--")
    plt.plot(history.history['val_' + 'accuracy'], "g--")
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.ylim((0.8, 1.00))
    plt.legend(['train', 'test'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], "r--")
    plt.plot(history.history['val_' + 'loss'], "g--")
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.ylim((0.0, 1.0))
    plt.legend(['train', 'test'], loc='best')

    plt.show()



#przygotwanie danych

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #mamy 10 klas dla danych

X_train, y_train = mnist_reader.load_mnist('fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion', kind='t10k')

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, len(class_names))
y_test = to_categorical(y_test, len(class_names))

#architektura modelu

model = Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=4,
                          verbose=1)

ModelCheck = ModelCheckpoint(filepath='best_model.h5', #zapisywac najlepszy model, bo zatrzyma sie raczej te 3/4 ustalone po najlepszym wyniku (gdy spada juz 3 razy)
                             monitor='val_loss',
                             save_best_only=True)

memory = model.fit(X_train,
                   y_train,
                   epochs=40,
                   # epoch – odpowiada za liczbę epok użytych do trenowania modelu – jedna epoka oznacza przejście całego zbioru przez sieć oraz powrót,
                   verbose=1,
                   # verbose – parametr, dzięki któremu można ustawić informacje jakie mają wyświetlać się podczas trenowania sieci,
                   batch_size=64,
                   # batch_size – odpowiada za zdefiniowanie ile rekordów (obserwacji) przechodzi na raz podczas pojedynczego przebiegu zanim nastąpi pierwsza aktualizacja wag parametrów,
                   validation_split=0.2,
                   # jak podzielić zbiór treningowy, na treningowy i testowy, jesli czesc chcemy trzymac do ost oceny
                   # *validation_data – ew przekazujemy nasz zbiór do walidacji.
                   callbacks=[EarlyStop, ModelCheck]
                   # monitoruje stratę na zbiorze testowym po zakończeniu każdej epoki. Jeśli strata nie maleje, wówczas trening sieci zostaje zatrzyman
                   )


model.evaluate(X_test, y_test, batch_size=32)

draw(memory)
