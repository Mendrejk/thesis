// #set text(font: "Montserrat")
#set text(
  font: "Satoshi",
  size: 12pt
)
#set par(justify: true)
#show link: underline
#set text(lang: "PL")

#import "@preview/big-todo:0.2.0": *


// Strona tytułowa
#set align(center)

#text(size: 22pt)[
Politechnika Wrocławska\
Wydział Informatyki i Telekomunikacji
]
#line(length: 100%)

#align(left)[
  Kierunek: *IST* \
  Specjalność: *PSI*
]

#block(spacing: 1em)#set text(font: "Atkinson Hyperlegible")

#text(size: 32pt)[
  PRACA DYPLOMOWA \
  MAGISTERSKA
]

*Analiza możliwości wykorzystania metod uczenia maszynowego w rekonstrukcji nagrań dźwiękowych*

#block(spacing: 1em)
Sebastian Łakomy-Pszon

#block(spacing: 1em)
Opiekun pracy

*Dr inż Maciej Walczyński*

#set align(left)

#pagebreak(weak: true)


= Algorytm sieci GAN - zastosowane rozwiązania

== Sieć neuronowa:

1. Architektura sieci:
   - Wykorzystano architekturę Generative Adversarial Network (GAN)
   - Generator oparty na strukturze enkoder-dekoder z połączeniami skip
   - Discriminator wykorzystujący architekturę konwolucyjną
   - Zastosowanie bloków rezydualnych w generatorze

2. Techniki normalizacji i regularyzacji:
   - Batch Normalization w generatorze
   - Spectral Normalization w warstwach konwolucyjnych
   - Gradient Penalty w treningu dyskryminatora
   - Instance Noise z mechanizmem annealing

3. Funkcje strat:
   - Adversarial Loss (Hinge Loss)
   - Content Loss (L1 lub L2)
   - Spectral Convergence Loss
   - Spectral Flatness Loss
   - Phase-Aware Loss
   - Multi-Resolution STFT Loss
   - Time-Frequency Loss
   - Signal-to-Noise Ratio (SNR) Loss
   - Perceptual Loss (z użyciem VGGish)

4. Optymalizacja:
   - Adam Optimizer z niestandardowymi parametrami beta
   - Gradient Accumulation
   - Dynamiczne dostosowywanie learning rate
   - Early Stopping

5. Przetwarzanie danych:
   - Praca na spektrogramach STFT
   - Normalizacja danych wejściowych i wyjściowych

6. Techniki augmentacji:
   - Dodawanie szumu winylowego do czystych nagrań

7. Monitorowanie i wizualizacja:
   - Zapisywanie checkpointów
   - Wizualizacja postępów treningu (straty, spektrogramy)
   - Logowanie iteracji i epok

8. Przetwarzanie audio:
   - Konwersja między domeną czasową a częstotliwościową
   - Manipulacja fazą i magnitudą spektrogramów

9. Ewaluacja:
   - Walidacja na oddzielnym zbiorze danych
   - Możliwość subiektywnej oceny przez odsłuch

10. Skalowalność:
    - Możliwość treningu na GPU z wykorzystaniem CUDA
    - Obsługa treningu na części danych (subset)

11. Inne techniki:
    - Feature Extractor (VGGish)
    - Wasserstein Loss

== Skrypty przygotowania danych:

1. Przetwarzanie audio:
   - Konwersja plików WAV do MP3 z wykorzystaniem biblioteki pydub
   - Segmentacja długich nagrań na 10-sekundowe fragmenty
   - Generowanie spektrogramów STFT (Short-Time Fourier Transform) z użyciem biblioteki librosa
   - Przekształcanie spektrogramów z powrotem na sygnał audio (inverse STFT)
   - Zastosowanie okna Hanna przy obliczaniu STFT

2. Augmentacja danych:
   - Symulacja efektu winylowego trzasku (vinyl crackle) do czystych nagrań
   - Generowanie różnych rodzajów szumów: trzaski, pęknięcia, zadrapania
   - Zastosowanie filtra pasmowo-przepustowego do symulacji charakterystyki częstotliwościowej płyt winylowych

3. Normalizacja i skalowanie danych:
   - Zastosowanie signed square root scaling do wartości STFT
   - Normalizacja amplitudy audio do zakresu 16-bitowego

4. Przetwarzanie równoległe:
   - Wykorzystanie ProcessPoolExecutor i ThreadPoolExecutor do równoległego przetwarzania plików
   - Dynamiczne dostosowywanie liczby procesów/wątków do dostępnych zasobów CPU

5. Zarządzanie pamięcią:
   - Ograniczanie użycia pamięci RAM poprzez przetwarzanie danych w mniejszych porcjach
   - Czyszczenie katalogów wyjściowych przed rozpoczęciem przetwarzania

6. Wizualizacja danych:
   - Generowanie wizualizacji spektrogramów STFT za pomocą biblioteki matplotlib
   - Tworzenie porównawczych wizualizacji dla różnych jakości audio

7. Organizacja danych:
   - Strukturyzacja danych w katalogach według typu (oryginalne, przekonwertowane, z efektem winylowym)
   - Spójne nazewnictwo plików dla łatwego łączenia powiązanych segmentów

8. Obsługa różnych formatów audio:
   - Praca z plikami WAV i MP3
   - Konwersja między formatami z zachowaniem metadanych (częstotliwość próbkowania, liczba kanałów)

9. Kontrola jakości:
   - Weryfikacja długości segmentów audio (dokładnie 10 sekund)
   - Sprawdzanie spójności danych między różnymi katalogami

10. Optymalizacja wydajności:
    - Wykorzystanie bibliotek numpy do szybkich operacji na tablicach
    - Zastosowanie generatorów i iteratorów do efektywnego przetwarzania dużych zbiorów danych

11. Interfejs użytkownika:
    - Implementacja interfejsu wiersza poleceń (CLI) z użyciem argparse
    - Interaktywne zapytania do użytkownika (np. czy wyczyścić istniejące pliki)

12. Monitorowanie postępu:
    - Wykorzystanie biblioteki tqdm do wyświetlania pasków postępu

13. Obsługa błędów:
    - Implementacja mechanizmów try-except do obsługi potencjalnych błędów podczas przetwarzania plików


== Szkielet projektu pracy magisterskiej

Spis treści

=== 1. Wstęp

  1.1. Tło historyczne nagrań dźwiękowych
  - Pierwsze nagranie dźwiękowe: Leon Scott, 1857 r., "Au Clair de la Lune"
  - Ewolucja technologii nagrywania: od fonografu Edisona po cyfrowe nośniki
  - Kluczowe momenty w historii rejestracji dźwięku
  - Wpływ rozwoju technologii na jakość i dostępność nagrań muzycznych
  
  1.2. Problematyka jakości historycznych nagrań
  - Ograniczenia wczesnych technologii nagrywania: wąskie pasmo, szumy, zniekształcenia
  - Wpływ warunków przechowywania na degradację nośników
  - Przykłady znaczących nagrań historycznych o niskiej jakości dźwięku
  - Wyzwania związane z odtwarzaniem i konserwacją starych nagrań
  
  1.3. Znaczenie rekonstrukcji nagrań muzycznych
  - Wartość kulturowa i historyczna archiwów muzycznych
  - Rola zrekonstruowanych nagrań w badaniach muzykologicznych
  - Wpływ jakości nagrań na percepcję i popularność utworów muzycznych
  - Ekonomiczne aspekty rekonstrukcji nagrań (np. rynek remasterów)
  
  1.4. Cel i zakres pracy
  - Główny cel: analiza możliwości wykorzystania metod uczenia maszynowego w rekonstrukcji nagrań dźwiękowych
  - Zakres badań: fokus na nagrania muzyczne, szczególnie historyczne
  - Kluczowe zagadnienia: usuwanie szumów, rozszerzanie pasma, uzupełnianie brakujących fragmentów
  - Planowane podejście: implementacja i analiza wybranych metod uczenia maszynowego, ze szczególnym uwzględnieniem sieci GAN
  - Oczekiwane rezultaty: ocena skuteczności proponowanych metod, wnioski dotyczące potencjału AI w rekonstrukcji nagrań

=== 2. Zagadnienie poprawy jakości sygnałów dźwiękowych

  2.1. Charakterystyka zniekształceń w nagraniach muzycznych
  - Główne typy zniekształceń występujących w nagraniach audio
  - Wpływ zniekształceń na percepcję muzyki i jej wartość artystyczną

  2.1.1. Szumy i trzaski
  - Źródła szumów: ograniczenia sprzętowe, zakłócenia elektromagnetyczne
  - Charakterystyka trzasków: przyczyny mechaniczne i elektroniczne
  - Wpływ szumów i trzasków na jakość odsłuchu
  
  2.1.2. Ograniczenia pasma częstotliwościowego
  - Historyczne ograniczenia w rejestrowaniu pełnego spektrum częstotliwości
  - Konsekwencje wąskiego pasma dla brzmienia instrumentów i wokalu
  - Znaczenie szerokiego pasma dla naturalności i pełni dźwięku
  
  2.1.3. Zniekształcenia nieliniowe
  - Definicja i przyczyny zniekształceń nieliniowych
  - Wpływ na harmoniczne i intermodulację w nagraniach
  - Trudności w korekcji zniekształceń nieliniowych
  
  2.2. Tradycyjne metody poprawy jakości nagrań
  - Ewolucja technik restauracji nagrań audio
  - Ograniczenia tradycyjnych metod
  
  2.2.1. Filtracja cyfrowa
  - Podstawowe typy filtrów: dolnoprzepustowe, górnoprzepustowe, pasmowe
  - Zastosowanie filtracji w redukcji szumów i korekcji częstotliwościowej
  - Wady i zalety filtracji cyfrowej w kontekście restauracji nagrań
  
  2.2.2. Remasterowanie
  - Definicja i cele procesu remasteringu
  - Typowe etapy remasteringu: normalizacja, kompresja, korekcja EQ
  - Kontrowersje wokół remasteringu: autentyczność vs. jakość dźwięku
  
  2.3. Wyzwania w rekonstrukcji historycznych nagrań muzycznych
  - Brak oryginalnych, wysokiej jakości źródeł dźwięku
  - Problemy z identyfikacją i separacją poszczególnych instrumentów
  - Zachowanie autentyczności brzmienia przy jednoczesnej poprawie jakości
  - Etyczne aspekty ingerencji w historyczne nagrania
  - Techniczne ograniczenia w odtwarzaniu oryginalnego brzmienia

=== 3. Metody sztucznej inteligencji w poprawie jakości nagrań dźwiękowych

  3.1. Przegląd technik uczenia maszynowego w przetwarzaniu dźwięku
  - Ewolucja zastosowań uczenia maszynowego w dziedzinie audio
  - Klasyfikacja głównych podejść: nadzorowane, nienadzorowane, półnadzorowane
  - Rola reprezentacji dźwięku w uczeniu maszynowym: spektrogramy, cechy MFCC, surowe próbki
  
  3.2. Sieci neuronowe w zadaniach audio
  - Ogólna charakterystyka sieci neuronowych w kontekście przetwarzania dźwięku
  - Porównanie efektywności różnych architektur w zadaniach audio
  
  3.2.1. Konwolucyjne sieci neuronowe (CNN)
  - Zasada działania CNN w analizie sygnałów audio
  - Zastosowania CNN w klasyfikacji dźwięków i rozpoznawaniu mowy
  - Adaptacje CNN do specyfiki danych dźwiękowych (np. dilated convolutions)
  
  3.2.2. Rekurencyjne sieci neuronowe (RNN)
  - Charakterystyka RNN i ich zdolność do modelowania sekwencji czasowych
  - Warianty RNN: LSTM, GRU i ich zastosowania w przetwarzaniu audio
  - Przykłady wykorzystania RNN w syntezie mowy i modelowaniu muzyki
  
  3.2.3. Autoenkodery
  - Koncepcja autoenkoderów i ich rola w redukcji wymiarowości danych audio
  - Zastosowania autoenkoderów w odszumianiu i kompresji sygnałów dźwiękowych
  - Warianty autoenkoderów: wariacyjne (VAE), splotowe (CAE) w kontekście audio
  
  3.3. Generatywne sieci przeciwstawne (GAN) w kontekście audio
  - Podstawowa idea GAN i jej adaptacja do domeny audio
  - Architektura GAN dla danych dźwiękowych: generator i dyskryminator
  - Zastosowania GAN w syntezie dźwięku, super-rozdzielczości audio i transferze stylu
  - Wyzwania związane z treningiem GAN dla sygnałów audio
  
  3.4. Modele dyfuzyjne w rekonstrukcji dźwięku
  - Wprowadzenie do koncepcji modeli dyfuzyjnych
  - Proces generacji dźwięku w modelach dyfuzyjnych: dodawanie i usuwanie szumu
  - Zastosowania modeli dyfuzyjnych w rekonstrukcji i syntezie audio
  - Porównanie modeli dyfuzyjnych z GAN w kontekście zadań audio
  - Aktualne osiągnięcia i perspektywy rozwoju modeli dyfuzyjnych w dziedzinie dźwięku

=== 4. Zastosowania metod sztucznej inteligencji w rekonstrukcji nagrań muzycznych

  - Ogólny przegląd praktycznych zastosowań AI w restauracji nagrań
  - Porównanie skuteczności metod AI z tradycyjnymi technikami
  - Wpływ postępu w dziedzinie AI na możliwości rekonstrukcji nagrań

  4.1. Usuwanie szumów i zakłóceń
  - Charakterystyka różnych typów szumów i zakłóceń w nagraniach muzycznych
  - Metody AI do identyfikacji i separacji szumów od sygnału muzycznego
  - Zastosowanie autoenkoderów i GAN w odszumianiu nagrań
  - Porównanie efektywności różnych architektur sieci w zadaniu usuwania szumów
  - Wyzwania związane z zachowaniem detali muzycznych podczas usuwania szumów
  
  4.2. Rozszerzanie pasma częstotliwościowego
  - Problematyka ograniczonego pasma w historycznych nagraniach
  - Techniki AI do estymacji i syntezy brakujących wysokich częstotliwości
  - Zastosowanie sieci GAN w super-rozdzielczości spektralnej
  - Metody oceny jakości rozszerzonego pasma częstotliwościowego
  - Etyczne aspekty dodawania nowych informacji do historycznych nagrań
  
  4.3. Uzupełnianie brakujących fragmentów
  - Przyczyny i charakterystyka ubytków w nagraniach muzycznych
  - Metody AI do interpolacji brakujących fragmentów audio
  - Zastosowanie modeli sekwencyjnych (RNN, LSTM) w przewidywaniu brakujących próbek
  - Wykorzystanie kontekstu muzycznego w rekonstrukcji ubytków
  - Ocena spójności muzycznej uzupełnionych fragmentów
  
  4.4. Poprawa jakości mocno skompresowanych plików audio
  - Wpływ kompresji stratnej na jakość nagrań muzycznych
  - Techniki AI do redukcji artefaktów kompresji i poprawy jakości dźwięku
  - Zastosowanie modeli GAN w odtwarzaniu detali utraconych podczas kompresji
  - Metody treningu sieci na parach nagrań przed i po kompresji
  - Wyzwania związane z generalizacją modeli na różne formaty kompresji

=== 5. Charakterystyka wybranej metody - sieci GAN w rekonstrukcji nagrań muzycznych

  - Wprowadzenie do koncepcji GAN w kontekście przetwarzania audio
  - Uzasadnienie wyboru GAN jako głównej metody do analizy w pracy
  
  5.1. Architektura sieci GAN dla zadań audio
  - Ogólna struktura GAN: generator i dyskryminator
  - Adaptacje architektury do specyfiki danych audio (np. 1D konwolucje)
  - Rola spektrogramów i reprezentacji czasowo-częstotliwościowych w GAN audio
  - Przykłady konkretnych implementacji GAN dla rekonstrukcji nagrań
  
  5.2. Proces uczenia sieci GAN
  - Zasada przeciwstawnego uczenia generatora i dyskryminatora
  - Specyfika treningu GAN dla danych audio: wyzwania i typowe problemy
  - Techniki stabilizacji procesu uczenia (np. normalizacja spektralna)
  - Strategie doboru hiperparametrów i zarządzania procesem treningowym
  
  5.3. Funkcje straty i metryki oceny jakości
  - Przegląd funkcji straty stosowanych w GAN audio (np. adversarial loss, reconstruction loss)
  - Obiektywne metryki oceny jakości rekonstrukcji audio (np. PESQ, STOI)
  - Subiektywne metody ewaluacji generowanych próbek dźwiękowych
  - Wyzwania związane z oceną jakości w kontekście historycznych nagrań
  
  5.4. Modyfikacje i rozszerzenia standardowej architektury GAN
  - Motywacja do wprowadzania modyfikacji w standardowej architekturze GAN
  - Przegląd najważniejszych wariantów GAN stosowanych w zadaniach audio
  
  5.4.1. Conditional GAN
  - Koncepcja warunkowego generowania w GAN
  - Zastosowania Conditional GAN w rekonstrukcji nagrań (np. kontrola parametrów rekonstrukcji)
  - Przykłady implementacji dla konkretnych zadań audio
  
  5.4.2. CycleGAN
  - Idea uczenia bez nadzoru w CycleGAN
  - Zastosowania CycleGAN w transferze stylu audio i konwersji głosu
  - Potencjał CycleGAN w rekonstrukcji nagrań bez par treningowych
  
  5.4.3. Progressive GAN
  - Koncepcja stopniowego zwiększania rozdzielczości w Progressive GAN
  - Adaptacje Progressive GAN do domeny audio
  - Zastosowania w generowaniu wysokiej jakości próbek dźwiękowych

=== 6. Implementacja i eksperymenty

  6.1. Opis zestawu danych
  - Charakterystyka wykorzystanych nagrań muzycznych
  - Proces przygotowania danych: konwersja WAV do MP3, segmentacja na 10-sekundowe fragmenty
  - Augmentacja danych: symulacja efektu winylowego trzasku, generowanie różnych rodzajów szumów
  - Normalizacja i skalowanie danych: signed square root scaling, normalizacja amplitudy
  
  6.2. Architektura proponowanego modelu
  - Szczegółowy opis architektury GAN
  - Generator: struktura enkoder-dekoder z połączeniami skip, bloki rezydualne
  - Discriminator: architektura konwolucyjna
  - Zastosowane techniki normalizacji: Batch Normalization, Spectral Normalization
  - Wykorzystanie spektrogramów STFT jako reprezentacji danych wejściowych
  
  6.3. Proces treningu i optymalizacji
  - Implementacja funkcji strat: Adversarial Loss, Content Loss, Spectral Convergence Loss, etc.
  - Optymalizacja: Adam Optimizer, gradient accumulation, dynamiczne dostosowywanie learning rate
  - Techniki regularyzacji: Gradient Penalty, Instance Noise z mechanizmem annealing
  - Monitorowanie i wizualizacja procesu treningu
  - Implementacja Early Stopping i zapisywanie checkpointów
  
  6.4. Charakterystyka kodu źródłowego
  - Struktura projektu i organizacja modułów
  - Kluczowe klasy i funkcje w implementacji sieci GAN
  - Wykorzystanie bibliotek: PyTorch, librosa, pydub
  - Mechanizmy przetwarzania równoległego i zarządzania pamięcią
  - Implementacja interfejsu wiersza poleceń (CLI) do obsługi skryptów
  
  6.5. Metodologia eksperymentów
  - Opis przeprowadzonych eksperymentów i ich celów
  - Metody ewaluacji: obiektywne metryki jakości audio, walidacja na oddzielnym zbiorze danych
  - Proces subiektywnej oceny przez odsłuch
  - Porównanie różnych wariantów modelu i konfiguracji hiperparametrów
  - Analiza wpływu poszczególnych komponentów (np. funkcji strat, technik normalizacji) na jakość rekonstrukcji

=== 7. Analiza wyników

  7.1. Ocena obiektywna
     - Omówienie zastosowanych metryk jakości dźwięku (np. PESQ, STOI)
     - Analiza wizualizacji spektrogramów STFT przed i po przetworzeniu
     - Obserwacja poprawy jakości spektrogramów w trakcie procesu uczenia
     - Dyskusja na temat niedostatecznej poprawy jakości do uzyskania użytecznych wyników audio
  
  7.2. Porównanie z innymi metodami
     - Zestawienie wyników z tradycyjnymi technikami przetwarzania sygnałów
     - Analiza porównawcza z innymi podejściami opartymi na uczeniu maszynowym
     - Omówienie potencjalnych przyczyn niższej skuteczności proponowanej metody
  
  7.3. Analiza procesu uczenia
     - Prezentacja krzywych uczenia dla generatora i dyskryminatora
     - Omówienie dynamiki treningu GAN i potencjalnych problemów (np. niestabilność, mode collapse)
     - Analiza wpływu różnych funkcji strat na jakość generowanych spektrogramów
  
  7.4. Studium przypadku: próba rekonstrukcji wybranego historycznego nagrania
     - Opis wybranego nagrania i jego charakterystycznych cech
     - Prezentacja wyników przetwarzania przez sieć GAN
     - Analiza przyczyn niepowodzenia w uzyskaniu słyszalnej poprawy jakości
  
  7.5. Dyskusja wyników
     - Omówienie głównych wyzwań napotkanych podczas eksperymentów
     - Analiza potencjalnych przyczyn niedostatecznej jakości rekonstrukcji audio
     - Propozycje możliwych ulepszeń i modyfikacji architektury sieci
     - Dyskusja na temat ograniczeń obecnego podejścia i potencjalnych kierunków dalszych badań

=== 8. Wnioski i perspektywy

  8.1. Podsumowanie osiągniętych rezultatów
  - Krótkie przypomnienie głównych celów pracy
  - Synteza kluczowych wyników eksperymentów
  - Ocena skuteczności proponowanej metody GAN w rekonstrukcji nagrań
  - Porównanie osiągniętych rezultatów z założeniami i oczekiwaniami
  
  8.2. Ograniczenia proponowanej metody
  - Identyfikacja głównych wyzwań i trudności napotkanych podczas badań
  - Analiza czynników ograniczających skuteczność rekonstrukcji (np. jakość danych treningowych, złożoność architektury)
  - Omówienie problemów specyficznych dla domeny audio w kontekście uczenia maszynowego
  - Rozważenie wpływu ograniczeń sprzętowych i obliczeniowych na jakość wyników
  
  8.3. Potencjalne kierunki dalszych badań
  - Propozycje modyfikacji i ulepszeń architektury sieci GAN
  - Sugestie dotyczące eksploracji alternatywnych technik uczenia maszynowego (np. modele dyfuzyjne)
  - Koncepcje integracji wiedzy dziedzinowej z zakresu przetwarzania sygnałów audio
  - Pomysły na rozszerzenie zakresu badań o inne gatunki muzyczne lub typy nagrań
  
  8.4. Implikacje dla przyszłości rekonstrukcji nagrań muzycznych
  - Ocena potencjału AI w kontekście ochrony dziedzictwa kulturowego
  - Rozważenie etycznych aspektów stosowania AI w rekonstrukcji historycznych nagrań
  - Prognoza rozwoju technologii AI w dziedzinie przetwarzania audio
  - Refleksja nad wpływem zaawansowanych technik rekonstrukcji na przemysł muzyczny i archiwizację

=== Bibliografia

Załączniki
   A. Spis symboli i skrótów
   B. Szczegółowe wyniki eksperymentów
   C. Fragmenty kodu źródłowego


= 1. Wstęp

== 1.1 Tło historyczne nagrań dźwiękowych

Historia rejestracji dźwięku sięga połowy XIX wieku, kiedy to w 1857 roku Édouard-Léon Scott de Martinville skonstruował fonoautograf - pierwsze urządzenie zdolne do zapisywania dźwięku @first-recorded-sound. Choć fonoautograf nie umożliwiał odtwarzania zarejestrowanych dźwięków, stanowił przełom w dziedzinie akustyki i zapoczątkował erę nagrań dźwiękowych. Pierwszym nagraniem uznawanym za możliwe do odtworzenia była francuska piosenka ludowa "Au Clair de la Lune", zarejestrowana przez Scotta w 1860 roku @first-recorded-sound.

Kolejnym kamieniem milowym w historii rejestracji dźwięku było wynalezienie fonografu przez Thomasa Edisona w 1877 roku. Urządzenie to nie tylko zapisywało dźwięk, ale również umożliwiało jego odtwarzanie, co otworzyło drogę do komercjalizacji nagrań dźwiękowych @edison-phonograph. W następnych dekadach technologia nagrywania ewoluowała, przechodząc przez etapy takie jak płyty gramofonowe, taśmy magnetyczne, aż po cyfrowe nośniki dźwięku @sound-recording-history.

Kluczowe momenty w historii rejestracji dźwięku obejmują:

1. 1888 - Wprowadzenie płyt gramofonowych przez Emile'a Berlinera @berliner-gramophone
2. 1920-1930 - Rozwój nagrań elektrycznych, znacząco poprawiających jakość dźwięku @electrical-recording
3. 1948 - Pojawienie się płyt długogrających (LP) @lp-record
4. 1963 - Wprowadzenie kaset kompaktowych przez Philips @compact-cassette
5. 1982 - Komercjalizacja płyt CD, rozpoczynająca erę cyfrową w muzyce @cd-introduction

Rozwój technologii nagrywania miał ogromny wpływ na jakość i dostępność nagrań muzycznych. Wczesne nagrania charakteryzowały się ograniczonym pasmem częstotliwości, wysokim poziomem szumów i zniekształceń @early-recording-limitations. Wraz z postępem technologicznym, jakość dźwięku stopniowo się poprawiała, osiągając szczyt w erze cyfrowej. Jednocześnie, ewolucja nośników dźwięku od płyt gramofonowych przez kasety magnetyczne po pliki cyfrowe, znacząco zwiększyła dostępność muzyki dla szerokiego grona odbiorców @music-accessibility.

Warto zauważyć, że mimo znacznego postępu technologicznego, wiele historycznych nagrań o ogromnej wartości kulturowej i artystycznej wciąż pozostaje w formie, która nie oddaje pełni ich oryginalnego brzmienia. Stwarza to potrzebę rozwoju zaawansowanych technik rekonstrukcji i restauracji nagrań, co stanowi jedno z głównych wyzwań współczesnej inżynierii dźwięku i muzykologii @audio-restoration-challenges.

#todo("[Potrzebne dodatkowe źródło dotyczące wyzwań w rekonstrukcji historycznych nagrań]", inline: false)


== 1.2. Problematyka jakości historycznych nagrań

Historyczne nagrania dźwiękowe, mimo ich ogromnej wartości kulturowej i artystycznej, często charakteryzują się niską jakością dźwięku, co stanowi istotne wyzwanie dla współczesnych słuchaczy i badaczy. Problematyka ta wynika z kilku kluczowych czynników.

Ograniczenia wczesnych technologii nagrywania stanowiły główną przeszkodę w uzyskiwaniu wysokiej jakości dźwięku. Wczesne urządzenia rejestrujące charakteryzowały się wąskim pasmem przenoszenia, co skutkowało utratą zarówno niskich, jak i wysokich częstotliwości @early-recording-limitations. Typowe pasmo przenoszenia dla nagrań z początku XX wieku wynosiło zaledwie 250-2500 Hz, co znacząco ograniczało pełnię brzmienia instrumentów i głosu ludzkiego @audio-bandwidth-history. Ponadto, pierwsze systemy nagrywające wprowadzały znaczne szumy i zniekształcenia do rejestrowanego materiału, co było spowodowane niedoskonałościami mechanicznymi i elektrycznymi ówczesnych urządzeń @noise-in-early-recordings.

Wpływ warunków przechowywania na degradację nośników jest kolejnym istotnym czynnikiem wpływającym na jakość historycznych nagrań. Nośniki analogowe, takie jak płyty gramofonowe czy taśmy magnetyczne, są szczególnie podatne na uszkodzenia fizyczne i chemiczne @analog-media-degradation. Ekspozycja na wilgoć, ekstremalne temperatury czy zanieczyszczenia powietrza może prowadzić do nieodwracalnych zmian w strukturze nośnika, co przekłada się na pogorszenie jakości odtwarzanego dźwięku @storage-conditions-impact. W przypadku taśm magnetycznych, zjawisko print-through, polegające na przenoszeniu sygnału magnetycznego między sąsiednimi warstwami taśmy, może wprowadzać dodatkowe zniekształcenia @print-through-effect.
Przykłady znaczących nagrań historycznych o niskiej jakości dźwięku są liczne i obejmują wiele kluczowych momentów w historii muzyki. Jednym z najbardziej znanych jest nagranie Johannesa Brahmsa wykonującego fragment swojego "Tańca węgierskiego nr 1" z 1889 roku, które jest najstarszym znanym nagraniem muzyki poważnej @brahms-recording. Nagranie to, mimo swojej ogromnej wartości historycznej, charakteryzuje się wysokim poziomem szumów i zniekształceń. Innym przykładem są wczesne nagrania bluesa, takie jak "Crazy Blues" Mamie Smith z 1920 roku, które pomimo przełomowego znaczenia dla rozwoju gatunku, cechują się ograniczonym pasmem częstotliwości i obecnością szumów tła @early-blues-recordings.

Wyzwania związane z odtwarzaniem i konserwacją starych nagrań są złożone i wymagają interdyscyplinarnego podejścia. Odtwarzanie historycznych nośników często wymaga specjalistycznego sprzętu, który sam w sobie może być trudny do utrzymania w dobrym stanie @playback-equipment-challenges. Proces digitalizacji, choć kluczowy dla zachowania dziedzictwa audio, niesie ze sobą ryzyko wprowadzenia nowych zniekształceń lub utraty subtelnych niuansów oryginalnego nagrania @music-digitization-challenges. Ponadto, konserwacja fizycznych nośników wymaga stworzenia odpowiednich warunków przechowywania, co może być kosztowne i logistycznie skomplikowane @preservation-storage-requirements.

Dodatkowo, etyczne aspekty restauracji nagrań historycznych stanowią przedmiot debaty w środowisku muzycznym i konserwatorskim. Pytanie o to, jak daleko można posunąć się w procesie cyfrowej rekonstrukcji bez naruszenia integralności oryginalnego dzieła, pozostaje otwarte @ethical-considerations-in-audio-restoration.
Problematyka jakości historycznych nagrań stanowi zatem nie tylko wyzwanie techniczne, ale również kulturowe i etyczne. Rozwój zaawansowanych technik rekonstrukcji audio, w tym metod opartych na sztucznej inteligencji, otwiera nowe możliwości w zakresie przywracania i zachowania dziedzictwa dźwiękowego, jednocześnie stawiając przed badaczami i konserwatorami nowe pytania dotyczące granic ingerencji w historyczny materiał @ai-in-audio-restoration.


== 1.3. Znaczenie rekonstrukcji nagrań muzycznych

Rekonstrukcja historycznych nagrań muzycznych odgrywa kluczową rolę w zachowaniu i promowaniu dziedzictwa kulturowego, oferując szereg korzyści zarówno dla badaczy, jak i dla szerszej publiczności.

Wartość kulturowa i historyczna archiwów muzycznych jest nieoceniona. Nagrania dźwiękowe stanowią unikalne świadectwo rozwoju muzyki, technik wykonawczych i zmian w stylistyce muzycznej na przestrzeni lat @cultural-value-of-music-archives. Rekonstrukcja tych nagrań pozwala na zachowanie i udostępnienie szerszemu gronu odbiorców dzieł, które w przeciwnym razie mogłyby zostać zapomniane lub stać się niedostępne ze względu na degradację nośników @preservation-of-audio-heritage. Ponadto, zrekonstruowane nagrania umożliwiają współczesnym słuchaczom doświadczenie wykonań legendarnych artystów w jakości zbliżonej do oryginalnej, co ma ogromne znaczenie dla zrozumienia historii muzyki i ewolucji stylów wykonawczych @historical-performance-practice.

Rola zrekonstruowanych nagrań w badaniach muzykologicznych jest fundamentalna. Wysokiej jakości rekonstrukcje pozwalają naukowcom na dokładną analizę technik wykonawczych, interpretacji i stylów muzycznych z przeszłości @musicological-research-methods. Umożliwiają one również badanie ewolucji praktyk wykonawczych oraz porównywanie różnych interpretacji tego samego utworu na przestrzeni lat @performance-practice-evolution. W przypadku kompozytorów, którzy sami wykonywali swoje dzieła, zrekonstruowane nagrania stanowią bezcenne źródło informacji o intencjach twórców @composer-performances.

Wpływ jakości nagrań na percepcję i popularność utworów muzycznych jest znaczący. Badania wskazują, że słuchacze są bardziej skłonni do pozytywnego odbioru i częstszego słuchania nagrań o wyższej jakości dźwięku @music-audio-quality-perception. Rekonstrukcja historycznych nagrań może przyczynić się do zwiększenia ich dostępności i atrakcyjności dla współczesnych odbiorców, potencjalnie prowadząc do odkrycia na nowo zapomnianych artystów lub utworów @rediscovery-of-forgotten-music. Ponadto, poprawa jakości dźwięku może pomóc w lepszym zrozumieniu i docenieniu niuansów wykonania, które mogły być wcześniej niezauważalne ze względu na ograniczenia techniczne oryginalnych nagrań @nuance-in-restored-recordings.

Ekonomiczne aspekty rekonstrukcji nagrań są również istotne. Rynek remasterów i zrekonstruowanych nagrań historycznych stanowi znaczący segment przemysłu muzycznego @remastered-recordings-market. Wydawnictwa specjalizujące się w tego typu projektach, takie jak "Deutsche Grammophon" czy "Naxos Historical", odnoszą sukcesy komercyjne, co świadczy o istnieniu popytu na wysokiej jakości wersje klasycznych nagrań @classical-music-reissues. Ponadto, rekonstrukcja nagrań może prowadzić do powstania nowych źródeł przychodów dla artystów lub ich spadkobierców, a także instytucji kulturalnych posiadających prawa do historycznych nagrań @revenue-from-restored-recordings.

#todo("[Potrzebne źródło dotyczące wpływu rekonstrukcji nagrań na edukację muzyczną]")

Warto również zauważyć, że rekonstrukcja nagrań muzycznych ma istotne znaczenie dla edukacji muzycznej. Zrekonstruowane nagrania historyczne mogą służyć jako cenne narzędzie dydaktyczne, umożliwiając studentom muzyki bezpośredni kontakt z wykonaniami wybitnych artystów z przeszłości i pomagając w zrozumieniu ewolucji stylów muzycznych.

Podsumowując, znaczenie rekonstrukcji nagrań muzycznych wykracza daleko poza samą poprawę jakości dźwięku. Jest to proces o fundamentalnym znaczeniu dla zachowania dziedzictwa kulturowego, wspierania badań naukowych, edukacji muzycznej oraz rozwoju przemysłu muzycznego. W miarę jak technologie rekonstrukcji dźwięku, w tym metody oparte na sztucznej inteligencji, stają się coraz bardziej zaawansowane, można oczekiwać, że ich rola w przywracaniu i promowaniu historycznych nagrań będzie nadal rosła, przynosząc korzyści zarówno dla świata nauki, jak i dla miłośników muzyki na całym świecie @future-of-audio-restoration.


== 1.4. Cel i zakres pracy

Głównym celem pracy jest dogłębna analiza i ocena potencjału metod uczenia maszynowego w dziedzinie rekonstrukcji nagrań dźwiękowych. Szczególny nacisk kładę na zbadanie efektywności zaawansowanych technik sztucznej inteligencji w przywracaniu jakości historycznym nagraniom muzycznym, koncentrując się na wyzwaniach, które tradycyjne metody przetwarzania sygnałów audio często pozostawiają nierozwiązane @ai-in-audio-restoration.

W ramach pracy skupię się na trzech kluczowych zagadnieniach:
1. Eliminacja szumów i zakłóceń typowych dla historycznych nagrań @noise-reduction-techniques.
2. Poszerzanie pasma częstotliwościowego w celu wzbogacenia brzmienia nagrań o ograniczonym paśmie @bandwidth-extension-methods.
3. Rekonstrukcja uszkodzonych fragmentów audio, co jest szczególnie istotne w przypadku wielu historycznych nagrań @audio-inpainting-techniques.

Podejście badawcze opiera się na implementacji i analizie wybranych metod uczenia maszynowego, z naciskiem na architekturę Generatywnych Sieci Przeciwstawnych (Generative Adversarial Networks - GAN) @gan-in-audio-processing. Wybór tej architektury wynika z jej udokumentowanej skuteczności w zadaniach generatywnych i rekonstrukcyjnych w innych powiązanych dziedzinach, takich jak przetwarzanie obrazów @gan-image-restoration.

W ramach badań planuję opracować zaawansowany model GAN, który będzie wykorzystywał strukturę enkoder-dekoder z połączeniami skip dla generatora oraz architekturę konwolucyjną dla dyskryminatora. Zamierzam zastosować szereg technik normalizacji i regularyzacji, takich jak Batch Normalization, Spectral Normalization czy Gradient Penalty, aby poprawić stabilność i wydajność treningu. Kluczowym elementem będzie wykorzystanie kompleksowego zestawu funkcji strat, obejmującego Adversarial Loss, Content Loss, oraz specjalistyczne funkcje straty dla domen audio, jak Spectral Convergence Loss czy Phase-Aware Loss. Planuję również zaimplementować zaawansowane techniki optymalizacji, w tym Adam Optimizer z niestandardowymi parametrami oraz dynamiczne dostosowywanie współczynnika uczenia.

Metodologia badawcza obejmuje kilka kluczowych etapów. Rozpocznę od przygotowania obszernego zestawu danych treningowych, składającego się z par nagrań oryginalnych i ich zdegradowanych wersji. Następnie zaimplementuję różnorodne warianty architektury GAN, dostosowane do specyfiki przetwarzania sygnałów audio. Proces treningu będzie wykorzystywał zaawansowane techniki, takie jak augmentacja danych czy przetwarzanie na spektrogramach STFT. Ostatnim etapem będzie wszechstronna ewaluacja wyników, łącząca wiele obiektywnych metryk jakości audio.

Oczekiwane rezultaty pracy obejmują kompleksową ocenę skuteczności proponowanych metod uczenia maszynowego w zadaniach rekonstrukcji nagrań audio, w zestawieniu z tradycyjnymi technikami przetwarzania sygnałów. Planuję przeprowadzić szczegółową analizę wpływu różnych komponentów architektury i parametrów modeli na jakość rekonstrukcji. Istotnym elementem będzie identyfikacja mocnych stron i ograniczeń metod opartych na AI w kontekście specyficznych wyzwań związanych z restauracją historycznych nagrań muzycznych. Na podstawie uzyskanych wyników, sformułuję wnioski dotyczące potencjału sztucznej inteligencji w dziedzinie rekonstrukcji nagrań, wraz z rekomendacjami dla przyszłych badań i zastosowań praktycznych.
 dziedzinie rekonstrukcji nagrań, wraz z rekomendacjami dla przyszłych badań i zastosowań praktycznych.

#todo("[Potrzebne źródło dotyczące etycznych aspektów wykorzystania AI w rekonstrukcji nagrań historycznych]")

Realizacja powyższych celów ma potencjał nie tylko do znaczącego wkładu w dziedzinę przetwarzania sygnałów audio i uczenia maszynowego, ale również do praktycznego zastosowania w procesach restauracji i zachowania dziedzictwa muzycznego @preservation-of-audio-heritage. Wyniki pracy mogą znaleźć zastosowanie w instytucjach kulturalnych, archiwach dźwiękowych oraz w przemyśle muzycznym, przyczyniając się do lepszego zachowania i udostępnienia cennych nagrań historycznych szerokiej publiczności.

#pagebreak(weak: true)

= 2. Zagadnienie poprawy jakości sygnałów dźwiękowych

== 2.1. Charakterystyka zniekształceń w nagraniach muzycznych

Zagadnienie poprawy jakości sygnałów dźwiękowych jest ściśle związane z charakterystyką zniekształceń występujących w nagraniach muzycznych. Zrozumienie natury tych zniekształceń jest kluczowe dla opracowania skutecznych metod ich redukcji lub eliminacji.

Główne typy zniekształceń występujących w nagraniach audio obejmują szereg problemów, które Szczotka @1 identyfikuje w swojej pracy, w tym szumy, brakujące dane, intermodulację i flutter. Szczególnie istotnym problemem, zwłaszcza w przypadku historycznych nagrań, jest ograniczenie pasma częstotliwościowego, co stanowi główny przedmiot badań w pracy nad BEHM-GAN @9. Wczesne systemy rejestracji dźwięku często były w stanie uchwycić jedynie wąski zakres częstotliwości, co prowadziło do utraty wielu detali dźwiękowych, szczególnie w zakresie wysokich i niskich tonów. Ponadto, jak wskazują badania nad rekonstrukcją mocno skompresowanych plików audio @11, kompresja może wprowadzać dodatkowe zniekształcenia, które znacząco wpływają na jakość dźwięku.

Zniekształcenia nieliniowe stanowią kolejną kategorię problemów, które mogą poważnie wpłynąć na jakość nagrania. Mogą one wynikać z niedoskonałości w procesie nagrywania, odtwarzania lub konwersji sygnału. Efektem tych zniekształceń może być wprowadzenie niepożądanych harmonicznych składowych lub intermodulacji, co prowadzi do zmiany charakteru dźwięku @analog-media-degradation.

W przypadku historycznych nagrań na nośnikach analogowych, takich jak płyty winylowe czy taśmy magnetyczne, często występują specyficzne rodzaje zniekształceń. Na przykład, efekt print-through w taśmach magnetycznych może prowadzić do pojawienia się echa lub przesłuchów między sąsiednimi warstwami taśmy @print-through-effect. Z kolei w przypadku płyt winylowych, charakterystyczne trzaski i szumy powierzchniowe są nieodłącznym elementem, który może znacząco wpływać na odbiór muzyki.

Wpływ tych zniekształceń na percepcję muzyki i jej wartość artystyczną jest znaczący. Badania pokazują, że jakość dźwięku ma istotny wpływ na to, jak słuchacze odbierają i oceniają muzykę @audio-quality-perception. Zniekształcenia mogą maskować subtelne niuanse wykonania, zmieniać barwę instrumentów czy głosu, a w skrajnych przypadkach całkowicie zniekształcać intencje artystyczne twórców.

W kontekście historycznych nagrań, zniekształcenia mogą stanowić barierę w pełnym docenieniu wartości artystycznej i kulturowej danego dzieła. Nawet niewielkie poprawy w jakości dźwięku mogą znacząco wpłynąć na odbiór i interpretację wykonania.

Jednocześnie warto zauważyć, że niektóre rodzaje zniekształceń, szczególnie te charakterystyczne dla określonych epok czy technologii nagrywania, mogą być postrzegane jako element autentyczności nagrania. To stawia przed procesem rekonstrukcji dźwięku wyzwanie znalezienia równowagi między poprawą jakości a zachowaniem historycznego charakteru nagrania @ethical-considerations-in-audio-restoration.

Zrozumienie charakterystyki zniekształceń w nagraniach muzycznych jest kluczowym krokiem w opracowaniu skutecznych metod ich redukcji. W kolejnych częściach pracy skupię się na tym, jak zaawansowane techniki uczenia maszynowego, w szczególności sieci GAN, mogą być wykorzystane do adresowania tych problemów, jednocześnie starając się zachować artystyczną integralność oryginalnych nagrań.

#pagebreak(weak: true)

= Przegląd literatury

== Spis literatury


+ #link("https://apd.usos.pwr.edu.pl/diplomas/8158/")[
  Projekt i implementacja wybranych algorytmów sztucznej inteligencji w procesie rekonstrukcji nagrań fonicznych
  [Przemysław Szczotka]
]

  Praca realizuje dwukierunkową konwolucyjno-rekurencyjną sieć neuronową (*BCRNN*) w celu usuwania zniekształceń takich jak szum, brakujące dane, intermodulacja i flutter. Praca ta pokazuje praktyczne zastosowanie metod uczenia maszynowego w przetwarzaniu sygnałów dźwiękowych i prezentuje wyniki eksperymentalne na bazie utworu Antoniego Vivaldiego "Cztery pory roku - Wiosna cz. I Allegro".

+ #link("https://apd.usos.pwr.edu.pl/diplomas/6496/")[
  Badanie jakości autorskich algorytmów cyfrowego przetwarzania sygnałów fonicznych
  [Michał Szubiński]
]

  #todo("[przeczytać i napisać opis]")
// TODO: tu opis jak dostanę pracę

+ #link("https://medium.com/illumination/restoring-and-reconstructing-old-and-damaged-compositions-using-ai-to-preserve-the-musical-heritage-b5eeb20b0039")[
  Restoring and reconstructing old and damaged compositions using AI to preserve the musical heritage
  [medium.com]
]

  Artykuł opisuje zastosowanie sztucznej inteligencji (AI) w *ochronie i rekonstrukcji dziedzictwa muzycznego*. Autorka pokazuje jak AI może pomóc w przywracaniu uszkodzonych kompozycji, rekonstruowaniu utraconych lub niedokończonych utworów, oraz jakie są etyczne i techniczne ograniczenia tego procesu. Autorka patrzy na problem z innej perspektywy, biorąc pod uwagę aspekty historyczne i kulturowe muzyki.

+ #link("https://arxiv.org/abs/2109.02692")[
  Machine Learning: Challenges, Limitations, and Compatibility for Audio Restoration Processes
  [Owen Casey; Rushit Dave; Naeem Seliya; Evelyn R Sowells Boone]
]

  Artykuł dotyczy wykorzystania sieci generatywnych przeciwnych (*GAN*) do poprawy jakości zdegradowanych i skompresowanych nagrań mowy. Autorzy opisują swoje próby zbudowania nowego modelu uczenia maszynowego na podstawie istniejącego projektu SEGAN, który miał na celu usunięcie szumów tła z dźwięku głosowego. Artykuł może być przydatny, ponieważ przedstawia wyzwania, ograniczenia i problemy zgodności związane z zastosowaniem uczenia maszynowego do zadań przetwarzania dźwięku.

+ #link("https://arxiv.org/pdf/2111.10891.pdf")[
  Active Restoration of Lost Audio Signals using Machine Learning and Latent Information
  [Zohra Adila Cheddad and Abbas Cheddad]
]

  Artykuł przedstawia nową metodę aktywnej rekonstrukcji utraconych sygnałów dźwiękowych z wykorzystaniem *steganografii, halftonowania i uczenia maszynowego*. Autorzy proponują połączenie tych trzech dziedzin nauki, aby osiągnąć lepszą jakość odtwarzania dźwięku i większą odporność na uszkodzenia danych. Ta metoda może być przydatna ponieważ pokazuje, jak wykorzystać ukryte informacje i zaawansowane modele uczenia maszynowego do naprawy uszkodzonych nagrań dźwiękowych.

+ #link("https://pdf.sciencedirectassets.com/271506/1-s2.0-S0957417423X00165/1-s2.0-S0957417423010886/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDZedhvUr59FORr8RtvmL0lu2uYf4bG6tQ%2B8GVOjGJ0sQIgT2v6IpTIaosVkQ9KP9H67V8Ea1URIgQbOTNMKJRrbqsqvAUIjv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDNsCB6Ym2qc8drSoeSqQBUmt72SlbfVbwcj5wWkCB%2F9yFzq6UZtkEAB%2B5%2BJfuvemTY6U9n85sDHwyfAotDB5l8M4qEU13FCmh75%2FRI9kGc2K%2B2Mx6BB0lRt1m4PVJvKm5vtOm0Eg737u3kC84RZRB1FHLtl9gCO8AAFVAIXFu0NxAWLHkt5rfrbijJqaqdC7E%2FSfiJfafOVZjwHUDs3DQu9vjttqaZk6Z0G7Ppi7rFgM8wYXonsuUzglxkZ8qFN9z14s9G4WREZY%2FtUNCIu2ncWHpIpH9Dr9OnREQlZjOXvkO4fzUaiTfyrtqgpqIbe8iBQVET%2BifI4jq0etAgrLb4MCWu07wzD0HbbzdGQ9pRtmL90TmBG9kn0JFW5QXIEF3ZCM9mCD7QQ6SrjtNC5KBAYCKG16UHE9X8pj12li%2FRC8o3fzUE6WW%2F3s6DnO9wUD9jd9LxZIAZfBVK5bR63gIJBkUiHjne6mS6YUZdef2hk6zdWS1jXk6vU1v3irCRSJ78wRqY10c7B8TqIk3lTgvtnrPgwMy0MQ%2BDrp%2FRsB5M2GW71N5Fu1%2F26njasJxak7upW19snzxqhl7idNcNlAY3tXxFgydRt5UWfJnWSR9b9RmIBGsg2%2F%2FsF9W2rWzABcynydFkecfWHWQCWyZb9h6%2F%2B9hoxKEh7gkXnII3RSBjc8l6fGZI0oVzxenI5Rnh2kTSBWAMc%2Bz6YZUnwC2iXv3FMK0FIKhjHpmPz7UEA%2BpMkeianQWipy2IE6TNgARAd2gnka4duZpg0E513sOxFSsRVwj7LGTP9CLXooXuzvIqWITJP4ANX5ddnpbagcZc9jDjr626uDRUHWz8IZVkSQaKhmVYMeMRGfEmEUkYu40JBZ96BzXUw6XR38J6%2FkrK3OMNOE%2Fq0GOrEB9SS%2BWP%2FdaKosYqHOC3Jqu0LmGfVRU3pnnoUIVJZMM2chmT9NCnDTLCWxyjQP0F119uwBLMg4k%2B1vIwsNIdho7JhZcZbjdvtQ3raK657PSedSZcrf5Ug3WglhWkX1%2Fd7Jce7%2BbNQScja7lcw%2F3817ZsulY5zFNuxW4Q%2BFv8J2lY8%2BtEEj3bwasLeWydlhdS6E4VFvichkB5r9SxDUyWfnAx5I38BdMFouTBOBL%2BLO78uN&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240204T141932Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYW6TGFZDO%2F20240204%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6e8cca352e098bc386e86f26c8663827bab036192b94025e145c178642563f16&hash=1d2503d5067a665dcb3eeb883b2c418a9aece00de36694bc72d2c2cc702f7004&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0957417423010886&tid=spdf-c5684dbf-6b64-47b6-a95a-47ddd3a31242&sid=59a2a7511e5301454e7af26-81491c79c29bgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0a0c5a5756545c025952&rr=85038c936da93512&cc=pl")[
  A deep learning framework for audio restoration using Convolutional/Deconvolutional Deep Autoencoders
  [Alberto Nogales, Santiago Donaher, Alvaro García-Tejedor]
]

  Ten artykuł przedstawia głęboką sieć neuronową opartą na *architekturze U-Net*, która służy do przywracania uszkodzonych nagrań dźwiękowych. Autorzy pokazują, że ich model osiąga lepsze wyniki niż klasyczne metody filtracji Wienera czy Wave-U-Net. Problem dotyczy zastosowania uczenia maszynowego do rekonstrukcji dźwięku, który jest istotny dla komunikacji i jakości nagrań.

+ #link("https://downloads.hindawi.com/journals/am/2018/3748141.pdf")[
  Research on Objective Evaluation of Recording Audio Restoration Based on Deep Learning Network
]

  Artykuł opisuje system oceny jakości nagrań historycznych oparty na *głębokiej sieci neuronowej z wykorzystaniem LSTM*. Autorzy porównują dwa rodzaje cech mowy: MFCC i GFCC i wykazują, że GFCC lepiej symuluje fizjologiczne właściwości ludzkiego ucha. Artykuł przedstawia *ciekawy system automatycznych ocen który może zostać zastosowany obok oceny eksperymentalnej na podstawie ankiety.*

+ #link("https://ieeexplore-1ieee-1org-1600oqgmc0074.han.bg.pwr.edu.pl/stamp/stamp.jsp?tp=&arnumber=9515219&tag=1")[
  Super-Resolution for Music Signals Using Generative Adversarial Networks [Jinhui Dai, Yue Zhang, Pengcheng Xie, Xinzhou Xu]
]

  Artykuł dotyczy zastosowania sieci generatywnych przeciwnych (*GAN*) do poprawy jakości *sygnałów muzycznych* poprzez zwiększenie ich rozdzielczości częstotliwościowej. Autorzy proponują metodę, która wykorzystuje spektogram logarytmiczny i fazę sygnału  i porównują ją z innymi podejściami opartymi na interpolacji - DNN i CNN. *Autorzy skupiają się na nagraniach o ograniczonym paśmie lub zniekształconych przez szum.*

+ #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9829821")[BEHM-GAN: Bandwidth Extension of Historical Music Using Generative Adversarial Networks]

  Sieć *GAN* do poprawy jakości *historycznych nagrań na pianinie*. Bardzo dobrze działa ale tylko do pianina. Potrzebne dalsze badania.

+ #link("https://www.epfl.ch/labs/mlo/wp-content/uploads/2022/10/crpmlcourse-paper1225.pdf")[Music Super-resolution with Spectral Flatness Loss and HiFi-GAN Discriminators
]

  Artykuł przedstawia ulepszenia w dziedzinie audio super-rozdzielczości przy użyciu *sieci GAN*. Autorzy integrują dyskryminator *HiFi-GAN* i wprowadzają *nową funkcję straty opartą na płaskości widmowej*. Badania wykazują poprawę w czasie treningu oraz potencjał do redukcji słyszalnego szumu, jednocześnie otwierając nowe możliwości w projektowaniu funkcji straty dla syntezy dźwięku wysokiej jakości.

+ #link("https://www.mdpi.com/2079-9292/10/11/1349#sec3dot3dot1-electronics-10-01349")[Stochastic Restoration of Heavily Compressed Musical Audio Using Generative Adversarial Networks]

  Artykuł prezentuje zastosowanie Generatywnych Sieci Przeciwstawnych (*GAN*) do poprawy jakości mocno *skompresowanych plików MP3*. Autorzy testują *stochastyczne i deterministyczne* generatory na plikach MP3 o różnych stopniach kompresji. Wyniki pokazują, że modele skutecznie poprawiają jakość dźwięku dla plików 16 i 32 kbit/s, *szczególnie w przypadku dźwięków perkusyjnych i wysokich częstotliwości*. Stochastyczny generator jest w stanie wytworzyć próbki bliższe oryginałowi niż generator deterministyczny. Badanie otwiera nowe możliwości w dziedzinie cyfrowej rekonstrukcji dźwięku.

+ #link("https://www.proquest.com/docview/2808565445/fulltextPDF/CEBFF7ED7C5D4870PQ/1?accountid=46407&sourcetype=Dissertations%20&%20Theses")[MACHINE LEARNING APPROACHES TO HISTORIC MUSIC RESTORATION]

  Praca przedstawia zastosowanie uczenia maszynowego do odrestaurowania historycznego *nagrania fortepianowego* Johannesa Brahmsa z 1889 roku. Wykorzystano metody *nienegatywnej faktoryzacji macierzy (NMF) oraz głębokich sieci neuronowych (DNN)* w połączeniu z *cyfrowym przetwarzaniem sygnałów*. Choć nie udało się przewyższyć wyników dotychczasowej restauracji, opracowane podejścia oferują pewne korzyści, w tym lepszą identyfikację początków nut oraz potencjał do szybszej wstępnej restauracji licznych historycznych nagrań.

+ #link("https://arxiv.org/pdf/2402.09821")[Diffusion Models for Audio Restoration]

  Artykuł przedstawia kompleksowe omówienie wykorzystania modeli dyfuzji w zadaniach związanych z restauracją dźwięku, takich jak poprawa jakości mowy i odtwarzanie muzyki. Autorzy omawiają różne techniki warunkowania modeli dyfuzji, które pozwalają na generowanie wysokiej jakości próbek dźwiękowych, jednocześnie umożliwiając wstrzykiwanie wiedzy dziedzinowej. Artykuł porusza również praktyczne wymagania związane z wykorzystaniem modeli dyfuzji, takie jak szybkość wnioskowania, przetwarzanie przyczynowe oraz odporność na niekorzystne warunki, przedstawiając różne podejścia do rozwiązania tych wyzwań, podkreślając potencjał modeli dyfuzyjnych w tworzeniu naturalnie brzmiących i niezawodnych algorytmów odtwarzania dźwięku.

#pagebreak(weak: true)

DO SPRAWDZENIA:

https://arxiv.org/abs/2010.04506
https://ieeexplore.ieee.org/abstract/document/9647041
https://www.epfl.ch/labs/mlo/wp-content/uploads/2022/10/crpmlcourse-paper1225.pdf
https://www.proquest.com/openview/1af6d6d8d818db037e018b9b30f3b027/1?pq-origsite=gscholar&cbl=18750&diss=y
https://www.mdpi.com/2079-9292/10/11/1349
https://ieeexplore.ieee.org/abstract/document/9746389


#pagebreak(weak: true)

== Podsumowanie literatury
Zebrane źródła skupiają się na sposobach restauracji, rekonstrukcji i oczyszczania szeroko rozumianych nagrań audio, w tym muzyki. W literaturze dominuje zastosowanie sieci GAN (Generative Adversarial Network) jako najlepiej radzących sobie z problemami natury dźwiękowej. Jest to kierunek w którym należy skupić się na dalszych badaniach. Jedno ze źródeł [7] wskazuje na metody oceniania tak nauczonych sieci co jest dobrym punktem wyjścia w celu analizy wyników pracy. Ogólnie można stwierdzić, że kierunek poprawy jakości nagrań audio cieszy się dużą popularnością wsród autorów publikacji i wreszcie wydaje się osiągalny dzięki licznym postępom w dziedzinie sztucznej inteligencji w ostatnich latach. 

#pagebreak(weak: true)

#bibliography("bibliography.yml")

#pagebreak(weak: true)

#todo_outline

