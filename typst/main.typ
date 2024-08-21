// #set text(font: "Montserrat")
#set text(
  font: "Satoshi",
  size: 12pt
)
#set par(justify: true)
#show link: underline
#set text(lang: "PL")

#set page(numbering: "1")

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

== 2.1.1. Szumy i trzaski
Szumy i trzaski stanowią jeden z najbardziej powszechnych problemów w historycznych nagraniach. Źródła szumów są różnorodne i obejmują ograniczenia sprzętowe, takie jak szum termiczny w elektronice, oraz zakłócenia elektromagnetyczne pochodzące z otoczenia lub samego sprzętu nagrywającego @noise-in-early-recordings. Charakterystyka trzasków jest często związana z przyczynami mechanicznymi, takimi jak uszkodzenia powierzchni płyt winylowych, lub elektronicznymi, wynikającymi z niedoskonałości w procesie zapisu lub odtwarzania.
Wpływ szumów i trzasków na jakość odsłuchu jest znaczący. Mogą one maskować subtelne detale muzyczne, zmniejszać dynamikę nagrania oraz powodować zmęczenie słuchacza. W skrajnych przypadkach, intensywne szumy lub częste trzaski mogą całkowicie zaburzyć odbiór muzyki, czyniąc nagranie trudnym lub niemożliwym do słuchania @audio-quality-perception.

== 2.1.2. Ograniczenia pasma częstotliwościowego
Historyczne ograniczenia w rejestrowaniu pełnego spektrum częstotliwości są jednym z kluczowych wyzwań w rekonstrukcji nagrań. Wczesne systemy nagrywania często były w stanie zarejestrować jedynie wąski zakres częstotliwości, typowo między 250 Hz a 2500 Hz @audio-bandwidth-history. To ograniczenie miało poważne konsekwencje dla brzmienia instrumentów i wokalu, prowadząc do utraty zarówno niskich tonów, nadających muzyce głębię i ciepło, jak i wysokich częstotliwości, odpowiedzialnych za klarowność i przestrzenność dźwięku.
Znaczenie szerokiego pasma dla naturalności i pełni dźwięku jest trudne do przecenienia. Współczesne badania pokazują, że ludzkie ucho jest zdolne do percepcji dźwięków w zakresie od około 20 Hz do 20 kHz, choć z wiekiem górna granica często się obniża. Pełne odtworzenie tego zakresu jest kluczowe dla realistycznego oddania brzmienia instrumentów i głosu ludzkiego. Rekonstrukcja szerokiego pasma częstotliwościowego w historycznych nagraniach stanowi zatem jedno z głównych zadań w procesie ich restauracji, co odzwierciedlają badania nad technikami takimi jak BEHM-GAN @9.

== 2.1.3. Zniekształcenia nieliniowe
Zniekształcenia nieliniowe stanowią szczególnie złożoną kategorię problemów w rekonstrukcji nagrań audio. Definiuje się je jako odstępstwa od idealnej, liniowej relacji między sygnałem wejściowym a wyjściowym w systemie audio. Przyczyny tych zniekształceń mogą być różnorodne, obejmując między innymi nasycenie magnetyczne w taśmach analogowych, nieliniową charakterystykę lamp elektronowych w starszym sprzęcie nagrywającym, czy też ograniczenia mechaniczne w przetwornikach @analog-media-degradation.
Wpływ zniekształceń nieliniowych na nagrania jest znaczący i często subtelny. Prowadzą one do powstania dodatkowych harmonicznych składowych dźwięku, które nie były obecne w oryginalnym sygnale, oraz do zjawiska intermodulacji, gdzie różne częstotliwości wejściowe generują nowe, niepożądane tony. W rezultacie, brzmienie instrumentów może ulec zmianie, a czystość i przejrzystość nagrania zostaje zaburzona. W niektórych przypadkach, zwłaszcza w muzyce elektronicznej czy rockowej, pewne formy zniekształceń nieliniowych mogą być celowo wprowadzane dla uzyskania pożądanego efektu artystycznego.
Korekcja zniekształceń nieliniowych stanowi jedno z największych wyzwań w procesie rekonstrukcji audio. W przeciwieństwie do zniekształceń liniowych, które można stosunkowo łatwo skorygować za pomocą filtrów, zniekształcenia nieliniowe wymagają bardziej zaawansowanych technik. Tradycyjne metody często okazują się niewystarczające, co skłania badaczy do poszukiwania rozwiązań opartych na uczeniu maszynowym, takich jak adaptacyjne modelowanie nieliniowości czy zastosowanie głębokich sieci neuronowych @11. Trudność polega na tym, że korekta tych zniekształceń wymaga precyzyjnego odtworzenia oryginalnego sygnału, co jest szczególnie skomplikowane w przypadku historycznych nagrań, gdzie brakuje referencyjnego materiału wysokiej jakości.

== 2.2. Tradycyjne metody poprawy jakości nagrań

Ewolucja technik restauracji nagrań audio przeszła znaczącą transformację od prostych metod analogowych do zaawansowanych technik cyfrowych. Początkowo, restauracja nagrań opierała się głównie na fizycznej konserwacji nośników i optymalizacji sprzętu odtwarzającego. Wraz z rozwojem technologii cyfrowej, pojawiły się nowe możliwości manipulacji sygnałem audio, co znacząco rozszerzyło arsenał narzędzi dostępnych dla inżynierów dźwięku @6. Nogales i inni w swojej pracy porównują efektywność klasycznych metod filtracji, takich jak filtr Wienera, z nowoczesnymi technikami głębokiego uczenia, ilustrując tę ewolucję.

Jednak tradycyjne metody, mimo swojej skuteczności w wielu przypadkach, mają pewne ograniczenia. Głównym problemem jest trudność w selektywnym usuwaniu szumów bez wpływu na oryginalny sygnał muzyczny. Ponadto, rekonstrukcja utraconych lub zniekształconych częstotliwości często prowadzi do artefaktów dźwiękowych, które mogą być równie niepożądane jak oryginalne zniekształcenia. Cheddad i Cheddad @5 w swoich badaniach nad aktywną rekonstrukcją utraconych sygnałów audio podkreślają te ograniczenia, proponując jednocześnie nowe podejścia uzupełniające klasyczne techniki restauracji.

=== 2.2.1. Filtracja cyfrowa

Filtracja cyfrowa stanowi podstawę wielu technik restauracji audio. Wyróżniamy trzy podstawowe typy filtrów: dolnoprzepustowe, górnoprzepustowe i pasmowe. Dai i inni @8 w swoich badaniach nad super-rozdzielczością sygnałów muzycznych pokazują, jak tradycyjne metody filtracji mogą być rozszerzone i ulepszone dzięki zastosowaniu uczenia maszynowego.

Zastosowanie filtracji w redukcji szumów polega na identyfikacji i selektywnym tłumieniu częstotliwości, w których dominuje szum. W korekcji częstotliwościowej, filtry są używane do wzmacniania lub osłabiania określonych zakresów częstotliwości, co pozwala na poprawę balansu tonalnego nagrania.

Wady filtracji cyfrowej obejmują ryzyko wprowadzenia artefaktów dźwiękowych, zwłaszcza przy agresywnym filtrowaniu, oraz potencjalną utratę subtelnych detali muzycznych. Zaletą jest natomiast precyzja i powtarzalność procesu, a także możliwość niedestrukcyjnej edycji.

=== 2.2.2. Remasterowanie

Remasterowanie to proces poprawy jakości istniejącego nagrania, często z wykorzystaniem nowoczesnych technologii cyfrowych. Celem remasteringu jest poprawa ogólnej jakości dźwięku, zwiększenie głośności do współczesnych standardów oraz dostosowanie brzmienia do współczesnych systemów odtwarzania.

Typowe etapy remasteringu obejmują normalizację, kompresję i korekcję EQ. Moliner i Välimäki @8 w swojej pracy nad BEHM-GAN pokazują, jak nowoczesne techniki mogą być wykorzystane do przezwyciężenia ograniczeń tradycyjnego remasteringu, szczególnie w kontekście rekonstrukcji wysokich częstotliwości w historycznych nagraniach muzycznych.

Kontrowersje wokół remasteringu często dotyczą konfliktu między zachowaniem autentyczności oryginalnego nagrania a dążeniem do poprawy jakości dźwięku. Lattner i Nistal @11 w swoich badaniach nad stochastyczną restauracją mocno skompresowanych plików audio pokazują, jak zaawansowane techniki mogą być wykorzystane do poprawy jakości nagrań bez utraty ich oryginalnego charakteru, co stanowi istotny głos w debacie o autentyczności vs. jakość dźwięku.

Mimo swoich ograniczeń, tradycyjne metody poprawy jakości nagrań wciąż odgrywają istotną rolę w procesie restauracji audio. Jednakże, rosnąca złożoność wyzwań związanych z restauracją historycznych nagrań skłania badaczy do poszukiwania bardziej zaawansowanych rozwiązań, w tym metod opartych na sztucznej inteligencji, które mogą przezwyciężyć niektóre z ograniczeń tradycyjnych technik.


== 2.3. Wyzwania w rekonstrukcji historycznych nagrań muzycznych

Proces rekonstrukcji historycznych nagrań muzycznych stawia przed badaczami szereg złożonych wyzwań, wymagających interdyscyplinarnego podejścia i zaawansowanych technik przetwarzania sygnałów.

Fundamentalnym problemem jest brak oryginalnych, wysokiej jakości źródeł dźwięku. Wiele historycznych nagrań przetrwało jedynie w formie znacznie zdegradowanej, często na nośnikach analogowych, które same uległy deterioracji @analog-media-degradation. Szczotka @1 zwraca uwagę, że niedobór niezakłóconych sygnałów referencyjnych komplikuje proces uczenia modeli rekonstrukcyjnych, zmuszając do opracowywania zaawansowanych metod symulacji degradacji dźwięku.

Identyfikacja i separacja poszczególnych instrumentów w nagraniach historycznych stanowi kolejne istotne wyzwanie. Dai i współpracownicy @8 podkreślają znaczenie tego aspektu, szczególnie w kontekście rekonstrukcji złożonych utworów orkiestrowych, gdzie ograniczenia wczesnych systemów nagrywania często prowadziły do nakładania się ścieżek instrumentalnych.

Kluczowym dylematem jest zachowanie autentyczności brzmienia przy jednoczesnej poprawie jakości. Moliner i Välimäki @9 akcentują potrzebę znalezienia równowagi między poprawą technicznej jakości dźwięku a utrzymaniem charakterystycznego, historycznego brzmienia nagrania. Zbyt agresywna ingerencja może prowadzić do utraty autentyczności i kontekstu historycznego.

Etyczne aspekty ingerencji w historyczne nagrania budzą kontrowersje w środowisku muzycznym i konserwatorskim. Lattner i Nistal @11 poruszają kwestię granic dopuszczalnej modyfikacji oryginalnego nagrania, argumentując za ostrożnym stosowaniem zaawansowanych technik rekonstrukcji.

Techniczne ograniczenia w odtwarzaniu oryginalnego brzmienia wynikają z fundamentalnych różnic między historycznymi a współczesnymi technologiami audio. Cheddad @5 zwracają uwagę na trudności w wiernym odtworzeniu charakterystyki akustycznej dawnych sal koncertowych czy specyfiki historycznych instrumentów.

Złożoność wyzwań związanych z rekonstrukcją historycznych nagrań muzycznych wymaga kompleksowego podejścia. Integracja zaawansowanych technik przetwarzania sygnałów, metod uczenia maszynowego, wiedzy muzykologicznej oraz refleksji etycznej jest kluczowa dla skutecznego rozwiązywania napotkanych problemów. Badania prowadzone przez Nogalesa i in. @6 wskazują na potrzebę ciągłego doskonalenia istniejących metod oraz opracowywania nowych rozwiązań. Przyszłość rekonstrukcji nagrań historycznych zależy od zdolności naukowców do tworzenia innowacyjnych technik, które będą w stanie sprostać unikalnym wymaganiom każdego historycznego dzieła muzycznego, zachowując jednocześnie jego autentyczność i wartość artystyczną.


== 3.1. Przegląd technik uczenia maszynowego w przetwarzaniu dźwięku

Rozwój metod uczenia maszynowego w ostatnich latach przyniósł znaczący postęp w dziedzinie przetwarzania i analizy sygnałów dźwiękowych. Techniki te znajdują coraz szersze zastosowanie w poprawie jakości nagrań, rekonstrukcji uszkodzonych fragmentów oraz ekstrakcji informacji z sygnałów audio.

=== 3.1.1. Ewolucja zastosowań uczenia maszynowego w dziedzinie audio

Początki wykorzystania uczenia maszynowego w przetwarzaniu dźwięku sięgają lat 90. XX wieku, kiedy to zaczęto stosować proste modele statystyczne do klasyfikacji gatunków muzycznych czy rozpoznawania mowy @4. Wraz z rozwojem mocy obliczeniowej komputerów oraz postępem w dziedzinie sztucznych sieci neuronowych, nastąpił gwałtowny wzrost zainteresowania tymi technikami w kontekście analizy i syntezy dźwięku.

Przełomowym momentem było zastosowanie głębokich sieci neuronowych, które umożliwiły modelowanie złożonych zależności w sygnałach audio. Badania wykazały, że głębokie sieci konwolucyjne potrafią skutecznie wyodrębniać cechy charakterystyczne dźwięków, co otworzyło drogę do bardziej zaawansowanych zastosowań, takich jak separacja źródeł dźwięku czy poprawa jakości nagrań.

W ostatnich latach coraz większą popularność zyskują modele generatywne, takie jak sieci GAN (Generative Adversarial Networks) czy modele dyfuzyjne, które umożliwiają nie tylko analizę, ale także syntezę wysokiej jakości sygnałów audio @8. Te zaawansowane techniki znajdują zastosowanie w rekonstrukcji uszkodzonych nagrań oraz rozszerzaniu pasma częstotliwości starych rejestracji dźwiękowych.

=== 3.1.2. Klasyfikacja głównych podejść: nadzorowane, nienadzorowane, półnadzorowane

W kontekście przetwarzania sygnałów audio można wyróżnić trzy główne podejścia do uczenia maszynowego:

a) Uczenie nadzorowane:
W tym podejściu model uczy się na podstawie par danych wejściowych i oczekiwanych wyników. W dziedzinie audio może to obejmować uczenie się mapowania między zaszumionymi a czystymi nagraniami w celu usuwania szumów, czy też klasyfikację instrumentów na podstawie oznaczonych próbek dźwiękowych. Przykładem zastosowania uczenia nadzorowanego jest praca Nogales A. i innych @6, w której autorzy wykorzystali konwolucyjne sieci neuronowe do rekonstrukcji uszkodzonych nagrań audio.

b) Uczenie nienadzorowane:
Techniki nienadzorowane skupiają się na odkrywaniu ukrytych struktur w danych bez korzystania z etykiet. W kontekście audio może to obejmować grupowanie podobnych dźwięków czy wyodrębnianie cech charakterystycznych bez uprzedniej wiedzy o ich znaczeniu.

c) Uczenie półnadzorowane:
To podejście łączy elementy uczenia nadzorowanego i nienadzorowanego, wykorzystując zarówno oznaczone, jak i nieoznaczone dane. Jest szczególnie przydatne w sytuacjach, gdy dostępna jest ograniczona ilość oznaczonych próbek, co często ma miejsce w przypadku historycznych nagrań audio.

=== 3.1.3. Rola reprezentacji dźwięku w uczeniu maszynowym: spektrogramy, cechy MFCC, surowe próbki

Wybór odpowiedniej reprezentacji dźwięku ma kluczowe znaczenie dla skuteczności modeli uczenia maszynowego w zadaniach przetwarzania audio.

a) Spektrogramy:
Przedstawiają rozkład częstotliwości sygnału w czasie, co pozwala na analizę zarówno cech czasowych, jak i częstotliwościowych. Spektrogramy są szczególnie przydatne w zadaniach takich jak separacja źródeł czy poprawa jakości nagrań. W pracy @8 autorzy wykorzystali spektrogramy logarytmiczne jako wejście do modelu GAN, osiągając dobre wyniki w zadaniu rozszerzania pasma częstotliwości nagrań muzycznych.

b) Cechy MFCC (Mel-Frequency Cepstral Coefficients):
Reprezentują charakterystykę widmową dźwięku w sposób zbliżony do ludzkiego systemu słuchowego. MFCC są często stosowane w zadaniach klasyfikacji i rozpoznawania mowy. Badania wykazały, że cechy MFCC mogą być skutecznie wykorzystywane w ocenie jakości rekonstrukcji nagrań historycznych.

c) Surowe próbki:
Niektóre modele, szczególnie te oparte na sieciach konwolucyjnych, mogą pracować bezpośrednio na surowych próbkach audio. Podejście to eliminuje potrzebę ręcznego projektowania cech, pozwalając modelowi na samodzielne odkrywanie istotnych wzorców w sygnale.

Wybór odpowiedniej reprezentacji zależy od specyfiki zadania oraz architektury modelu. Coraz częściej stosuje się też podejścia hybrydowe, łączące różne reprezentacje w celu uzyskania lepszych wyników.

Techniki uczenia maszynowego oferują szerokie spektrum możliwości w dziedzinie przetwarzania i poprawy jakości sygnałów audio. Ewolucja tych metod, od prostych modeli statystycznych po zaawansowane sieci generatywne, umożliwia rozwiązywanie coraz bardziej złożonych problemów związanych z rekonstrukcją i poprawą jakości nagrań dźwiękowych. W kontekście przetwarzania sygnałów audio kluczowe znaczenie ma odpowiedni dobór podejścia (nadzorowane, nienadzorowane lub półnadzorowane) oraz reprezentacji dźwięku. Właściwe decyzje w tym zakresie pozwalają na optymalne wykorzystanie potencjału uczenia maszynowego, co przekłada się na skuteczność i efektywność opracowywanych rozwiązań. Postęp w tej dziedzinie otwiera nowe możliwości w zakresie zachowania i odtwarzania dziedzictwa kulturowego, jakim są historyczne nagrania dźwiękowe.


== 3.2. Sieci neuronowe w zadaniach audio

Sieci neuronowe stały się fundamentalnym narzędziem w przetwarzaniu sygnałów dźwiękowych, oferując niezrównaną elastyczność i zdolność do modelowania złożonych zależności. Ich adaptacyjna natura pozwala na automatyczne wyodrębnianie istotnych cech z surowych danych audio, co czyni je niezwykle skutecznymi w szerokiej gamie zastosowań - od klasyfikacji dźwięków po zaawansowaną syntezę mowy.

Różnorodność architektur sieci neuronowych pozwala na dobór optymalnego rozwiązania do specyfiki danego zadania audio. Konwolucyjne sieci neuronowe (CNN) wykazują szczególną skuteczność w analizie lokalnych wzorców w spektrogramach, podczas gdy rekurencyjne sieci neuronowe (RNN) doskonale radzą sobie z modelowaniem długoterminowych zależności czasowych. Autoenkodery z kolei znajdują zastosowanie w kompresji i odszumianiu sygnałów, oferując możliwość redukcji wymiarowości przy zachowaniu kluczowych cech dźwięku.

Efektywność poszczególnych architektur może się znacząco różnić w zależności od konkretnego zadania. Badania empiryczne wskazują, że hybrydowe podejścia, łączące zalety różnych typów sieci, często prowadzą do najlepszych rezultatów w złożonych scenariuszach przetwarzania audio.

=== 3.2.1. Konwolucyjne sieci neuronowe (CNN)

Konwolucyjne sieci neuronowe zrewolucjonizowały sposób, w jaki analizujemy sygnały audio. Ich unikalna architektura, inspirowana biologicznym systemem wzrokowym, okazała się niezwykle skuteczna w wyodrębnianiu hierarchicznych cech z reprezentacji czasowo-częstotliwościowych dźwięku.

W kontekście analizy audio, CNN operują najczęściej na spektrogramach, traktując je jako dwuwymiarowe "obrazy" dźwięku. Warstwy konwolucyjne działają jak filtry, wyodrębniając lokalne wzorce spektralne, które mogą odpowiadać konkretnym cechom akustycznym, takim jak akordy, formanty czy charakterystyki instrumentów.

Klasyfikacja dźwięków i rozpoznawanie mowy to obszary, w których sieci CNN wykazują szczególną skuteczność. W zadaniach identyfikacji gatunków muzycznych czy detekcji słów kluczowych, sieci te potrafią automatycznie nauczyć się rozpoznawać istotne cechy spektralne, często przewyższając tradycyjne metody oparte na ręcznie projektowanych cechach.

Adaptacje CNN do specyfiki danych dźwiękowych obejmują m.in. zastosowanie dilated convolutions. Ta technika pozwala na zwiększenie pola recepcyjnego sieci bez zwiększania liczby parametrów, co jest szczególnie przydatne w modelowaniu długoterminowych zależności czasowych w sygnałach audio. Dilated CNN znalazły zastosowanie m.in. w generowaniu dźwięku w czasie rzeczywistym.

=== 3.2.2. Rekurencyjne sieci neuronowe (RNN)

Rekurencyjne sieci neuronowe wyróżniają się zdolnością do przetwarzania sekwencji danych, co czyni je naturalnym wyborem do analizy sygnałów audio. Ich architektura, oparta na pętlach sprzężenia zwrotnego, pozwala na uwzględnienie kontekstu czasowego w przetwarzaniu dźwięku, co jest kluczowe w wielu zadaniach, takich jak modelowanie muzyki czy rozpoznawanie mowy ciągłej.

LSTM (Long Short-Term Memory) i GRU (Gated Recurrent Unit) to popularni "następcy" klasycznych RNN, którzy rozwiązują problem zanikającego gradientu. Te zaawansowane jednostki rekurencyjne potrafią efektywnie przetwarzać długie sekwencje audio, zachowując informacje o odległych zależnościach czasowych.

W syntezie mowy, modele oparte na LSTM wykazały się zdolnością do generowania naturalnie brzmiących wypowiedzi, uwzględniających niuanse prozodyczne. W dziedzinie modelowania muzyki, sieci rekurencyjne znalazły zastosowanie w generowaniu sekwencji akordów czy komponowaniu melodii, potrafiąc uchwycić złożone struktury harmoniczne i rytmiczne.

=== 3.2.3. Autoenkodery

Autoenkodery to fascynująca klasa sieci neuronowych, której głównym zadaniem jest nauczenie się efektywnej, skompresowanej reprezentacji danych wejściowych. W kontekście audio, ta zdolność do redukcji wymiarowości otwiera szereg możliwości - od kompresji sygnałów po zaawansowane techniki odszumiania.

Klasyczny autoenkoder składa się z enkodera, który "ściska" dane wejściowe do niższego wymiaru, oraz dekodera, który próbuje odtworzyć oryginalne dane z tej skompresowanej reprezentacji. W zastosowaniach audio, autoenkodery mogą nauczyć się reprezentacji, które zachowują kluczowe cechy dźwięku, jednocześnie eliminując szum czy niepożądane artefakty.

Wariacyjne autoenkodery (VAE) idą o krok dalej, wprowadzając element losowości do procesu kodowania. Ta cecha czyni je szczególnie przydatnymi w generowaniu nowych, unikalnych dźwięków, zachowujących charakterystykę danych treningowych. VAE znalazły zastosowanie m.in. w syntezie mowy i efektów dźwiękowych.

Splotowe autoenkodery (CAE) łączą zalety autoenkoderów i CNN, co czyni je skutecznymi w zadaniach związanych z przetwarzaniem spektrogramów. Ich zdolność do wyodrębniania lokalnych cech spektralnych przy jednoczesnej redukcji wymiarowości sprawia, że są cennym narzędziem w odszumianiu i restauracji nagrań audio.

== 3.3. Generatywne sieci przeciwstawne (GAN) w kontekście audio

Generatywne sieci przeciwstawne (GAN) to innowacyjna architektura uczenia maszynowego, która zrewolucjonizowała podejście do generacji i przetwarzania danych, w tym sygnałów audio. Podstawowa idea GAN opiera się na "rywalizacji" dwóch sieci neuronowych: generatora, który tworzy nowe dane, oraz dyskryminatora, który ocenia ich autentyczność. Ta koncepcja, początkowo opracowana dla obrazów, została z powodzeniem zaadaptowana do domeny audio, otwierając nowe możliwości w syntezie i manipulacji dźwiękiem.

W kontekście danych dźwiękowych, architektura GAN wymaga specyficznego podejścia. Generator często pracuje na reprezentacjach czasowo-częstotliwościowych, takich jak spektrogramy, tworząc nowe "obrazy" dźwięku. Dyskryminator z kolei analizuje te reprezentacje, ucząc się rozróżniać między autentycznymi a wygenerowanymi próbkami. Kluczowym wyzwaniem jest zapewnienie, aby wygenerowane spektrogramy były nie tylko realistyczne wizualnie, ale także przekładały się na spójne i naturalne brzmienia po konwersji z powrotem do domeny czasowej.

Zastosowania GAN w dziedzinie audio są niezwykle różnorodne. W syntezie dźwięku, sieci te potrafią generować realistyczne efekty dźwiękowe czy nawet całe utwory muzyczne, naśladując style konkretnych artystów. W zadaniach super-rozdzielczości audio, sieci GAN wykazują imponującą zdolność do rekonstrukcji wysokich częstotliwości w nagraniach o ograniczonym paśmie, co znajduje zastosowanie w restauracji historycznych nagrań. Transfer stylu audio, inspirowany podobnymi technikami w przetwarzaniu obrazów, pozwala na przenoszenie charakterystyk brzmieniowych między różnymi nagraniami, otwierając fascynujące możliwości w produkcji muzycznej.

Trening GAN dla sygnałów audio niesie ze sobą specyficzne wyzwania. Niestabilność treningu, charakterystyczna dla GAN, jest szczególnie problematyczna w domenie audio, gdzie nawet drobne artefakty mogą znacząco wpłynąć na jakość percepcyjną. Projektowanie odpowiednich funkcji straty, które uwzględniają specyfikę ludzkiego słuchu, stanowi kolejne wyzwanie. Ponadto, zapewnienie spójności fazowej w generowanych spektrogramach wymaga dodatkowych technik, takich jak wykorzystanie informacji o fazie lub bezpośrednie generowanie w domenie czasowej.

== 3.4. Modele dyfuzyjne w rekonstrukcji dźwięku

Modele dyfuzyjne reprezentują nowatorskie podejście do generacji danych, które w ostatnich latach zyskało ogromną popularność w dziedzinie przetwarzania dźwięku. U podstaw tej koncepcji leży idea stopniowego dodawania szumu do danych, a następnie uczenia się procesu odwrotnego - usuwania szumu, co prowadzi do generacji nowych, wysokiej jakości próbek.

Proces generacji dźwięku w modelach dyfuzyjnych można podzielić na dwa etapy. W pierwszym, zwanym procesem forward, do oryginalnego sygnału audio stopniowo dodawany jest szum gaussowski, aż do otrzymania czystego szumu. W drugim etapie, zwanym procesem reverse, model uczy się krok po kroku usuwać ten szum, rozpoczynając od losowej próbki szumu i stopniowo przekształcając ją w realistyczny sygnał audio. Ta unikalna architektura pozwala na generację dźwięku o wysokiej jakości i szczegółowości.

Zastosowania modeli dyfuzyjnych w rekonstrukcji i syntezie audio są obiecujące. W zadaniach rekonstrukcji uszkodzonych nagrań, modele te wykazują zdolność do "wypełniania" brakujących fragmentów w sposób spójny z resztą nagrania. W syntezie mowy, modele dyfuzyjne potrafią generować niezwykle naturalne i ekspresyjne wypowiedzi, uwzględniając subtelne niuanse prozodyczne.

W porównaniu z GAN, modele dyfuzyjne oferują kilka istotnych zalet w kontekście zadań audio. Przede wszystkim, ich trening jest bardziej stabilny i przewidywalny, co przekłada się na konsekwentnie wysoką jakość generowanych próbek. Modele dyfuzyjne wykazują również lepszą zdolność do modelowania różnorodności danych, unikając problemu "mode collapse" charakterystycznego dla GAN. Jednakże, kosztem tych zalet jest zazwyczaj dłuższy czas generacji, co może ograniczać ich zastosowanie w aplikacjach czasu rzeczywistego.

Aktualne osiągnięcia w dziedzinie modeli dyfuzyjnych dla dźwięku są imponujące. Modele takie jak WaveGrad czy DiffWave demonstrują wysoką jakość w syntezie mowy, często przewyższając modele autoregresyjne. W dziedzinie muzyki, modele dyfuzyjne pokazują rezultaty w generacji instrumentalnej i wokalnej, zachowując niezwykłą szczegółowość brzmienia.

Eksplorowane są techniki łączenia modeli dyfuzyjnych z innymi architekturami, takimi jak transformery, w celu lepszego modelowania długoterminowych zależności w sygnałach audio. Rosnące zainteresowanie multimodalnych modeli dyfuzyjnych otwiera możliwości syntezy audio skorelowanej z innymi modalnościami, takimi jak obraz czy tekst.

Zarówno GAN, jak i modele dyfuzyjne reprezentują przełomowe podejścia w dziedzinie generacji i rekonstrukcji dźwięku. Każda z tych technik oferuje unikalne zalety. Dalszy rozwój tych metod niewątpliwie przyczyni się do postępu w takich dziedzinach jak restauracja historycznych nagrań, synteza mowy czy produkcja muzyczna, otwierając nowe horyzonty w przetwarzaniu i generacji sygnałów audio.


== 4. Zastosowania metod sztucznej inteligencji w rekonstrukcji nagrań muzycznych

=== Ogólny przegląd praktycznych zastosowań AI w restauracji nagrań

Zastosowanie sztucznej inteligencji (AI) w rekonstrukcji nagrań muzycznych stale zyskuje na popularności. Tradycyjne techniki restauracji, jak filtry analogowe i cyfrowe, miały swoje ograniczenia, szczególnie w kontekście skomplikowanych sygnałów muzycznych. Nowoczesne metody AI, w tym głębokie uczenie i generatywne sieci przeciwstawne (GAN), oferują nowe możliwości w przywracaniu uszkodzonych i zdegradowanych nagrań muzycznych, poprawiając ich jakość w sposób, który wcześniej nie był możliwy.

Przykładowo, praca Dai et al. pokazuje, jak sieci GAN mogą być wykorzystane do poprawy rozdzielczości sygnałów muzycznych, co prowadzi do bardziej precyzyjnej i szczegółowej rekonstrukcji dźwięku @8. Z kolei badania przedstawione przez Nogales et al. wykorzystują głębokie autoenkodery do przywracania jakości nagrań, przewyższając tradycyjne metody, takie jak filtracja Wienera @6.

=== Porównanie skuteczności metod AI z tradycyjnymi technikami

Tradycyjne metody rekonstrukcji nagrań muzycznych, takie jak filtry Wienera czy metody interpolacji oparte na DSP, są powszechnie stosowane, ale ich skuteczność jest ograniczona. Wprowadzenie technik AI, w szczególności głębokich sieci neuronowych, znacząco poprawiło jakość odtwarzania i rekonstrukcji nagrań.

Przykładem jest zastosowanie GAN do poprawy jakości mocno skompresowanych plików MP3. Artykuł z MDPI pokazuje, jak stochastyczne generatory oparte na GAN są w stanie wytworzyć próbki bliższe oryginałowi niż tradycyjne metody DSP, szczególnie w przypadku dźwięków perkusyjnych i wysokich częstotliwości @11. Dodatkowo, metody takie jak nienegatywna faktoryzacja macierzy (NMF) i głębokie sieci neuronowe (DNN) zostały zastosowane do odrestaurowania historycznych nagrań fortepianowych, jak pokazuje praca na temat rekonstrukcji nagrania Johannesa Brahmsa z 1889 roku @4.

=== Wpływ postępu w dziedzinie AI na możliwości rekonstrukcji nagrań

Postęp w dziedzinie AI, a zwłaszcza rozwój modeli dyfuzyjnych i GAN, otworzył nowe możliwości w rekonstrukcji nagrań muzycznych. Modele te pozwalają na generowanie dźwięku o wysokiej jakości, nawet z uszkodzonych i silnie skompresowanych źródeł. Artykuł na temat modeli dyfuzyjnych dla restauracji dźwięku przedstawia kompleksowe omówienie tego tematu, podkreślając ich zdolność do generowania naturalnie brzmiących próbek dźwiękowych @13.


== 4.1. Usuwanie szumów i zakłóceń

Usuwanie szumów i zakłóceń z nagrań muzycznych stanowi kluczowe wyzwanie w procesie rekonstrukcji dźwięku, szczególnie w kontekście metod opartych na sztucznej inteligencji. Nagrania muzyczne mogą być narażone na różnorodne typy szumów, takie jak szumy tła, impulsowe zakłócenia oraz artefakty powstałe podczas konwersji analogowo-cyfrowej. W celu ich skutecznego usunięcia, konieczne jest zrozumienie charakterystyki każdego z tych szumów, a także ich wpływu na jakość odbioru dźwięku przez słuchacza.

W ostatnich latach, metody oparte na sztucznej inteligencji, w tym sieci neuronowe, zyskały na popularności w kontekście identyfikacji i separacji szumów od sygnału muzycznego. Zastosowanie autoenkoderów oraz sieci GAN okazało się szczególnie efektywne w odszumianiu nagrań, co potwierdzają liczne badania @14. Autoenkodery, ze względu na swoją zdolność do kompresji danych i ich rekonstrukcji, umożliwiają wyodrębnienie istotnych cech sygnału, a jednocześnie eliminację niepożądanych szumów. Z kolei sieci GAN, które składają się z generatora i dyskryminatora, pozwalają na generowanie bardziej realistycznych rekonstrukcji sygnału dźwiękowego, dzięki czemu możliwe jest zachowanie większej ilości detali muzycznych podczas usuwania szumów @14.

Porównanie efektywności różnych architektur sieci neuronowych w zadaniu usuwania szumów wykazało, że tradycyjne metody oparte na filtracji spektralnej ustępują nowoczesnym podejściom opartym na głębokim uczeniu się. Przykładem może być zastosowanie bloków rezydualnych oraz technik normalizacji w architekturze sieci, co prowadzi do znaczącej poprawy jakości odszumionego dźwięku @14.

Niemniej jednak, wyzwania związane z zachowaniem detali muzycznych podczas usuwania szumów pozostają istotnym problemem. Głębokie sieci uczące się często mają tendencję do usuwania nie tylko szumów, ale również subtelnych niuansów muzycznych, co może prowadzić do utraty pierwotnego charakteru nagrania. Aby zminimalizować ten efekt, stosowane są zaawansowane funkcje strat, takie jak Perceptual Loss czy Signal-to-Noise Ratio Loss, które pomagają w zachowaniu jak największej ilości oryginalnych detali dźwiękowych @15.


== 4.2. Rozszerzanie pasma częstotliwościowego

Rozszerzanie pasma częstotliwościowego w historycznych nagraniach stanowi istotne wyzwanie technologiczne i badawcze, mające na celu poprawę jakości dźwięku przy zachowaniu integralności oryginalnego materiału.

=== Problematyka ograniczonego pasma w historycznych nagraniach

Historyczne nagrania, z uwagi na ograniczenia technologiczne ówczesnych systemów rejestracji dźwięku, często charakteryzują się ograniczonym pasmem przenoszenia, co prowadzi do utraty wyższych częstotliwości i w rezultacie zubożenia jakości dźwięku. Tego typu nagrania są zwykle poddawane cyfryzacji, a następnie obróbce mającej na celu odzyskanie jak największej ilości utraconej informacji. Rozszerzanie pasma częstotliwościowego staje się tutaj kluczowym narzędziem, które umożliwia przywrócenie pełniejszego brzmienia nagrania, a co za tym idzie, zbliżenie się do oryginalnego zamysłu artystycznego twórcy.

=== Techniki AI do estymacji i syntezy brakujących wysokich częstotliwości

Zastosowanie sztucznej inteligencji, w szczególności technik uczenia maszynowego, przyniosło nowe możliwości w zakresie rekonstrukcji brakujących informacji w historycznych nagraniach. Przykładem tego jest metoda Blind Audio Bandwidth Extension (BABE), która wykorzystuje model dyfuzyjny do estymacji brakujących wysokich częstotliwości w nagraniach o ograniczonym paśmie przenoszenia. Model ten, działający w tzw. trybie zero-shot, pozwala na realistyczne odtworzenie utraconych części spektrum częstotliwości bez konieczności znajomości szczegółów degradacji sygnału @16. Testy subiektywne potwierdziły, że zastosowanie BABE znacząco poprawia jakość dźwięku w nagraniach historycznych @16.

=== Zastosowanie sieci GAN w super-rozdzielczości spektralnej

Sieci generatywne (GAN) znalazły szerokie zastosowanie w przetwarzaniu dźwięku, w tym w rozszerzaniu pasma częstotliwościowego. Metoda BEHM-GAN wykorzystuje sieci GAN do rozszerzania pasma częstotliwościowego w nagraniach muzycznych z początku XX wieku. Zastosowanie GAN pozwala na realistyczną syntezę brakujących wysokich częstotliwości, co przekłada się na znaczną poprawę percepcyjnej jakości dźwięku @9.

=== Metody oceny jakości rozszerzonego pasma częstotliwościowego

Ocena jakości dźwięku po zastosowaniu technik rozszerzania pasma częstotliwościowego jest kluczowym etapem procesu. W przypadku historycznych nagrań ocena ta jest szczególnie istotna, ponieważ dodanie nowych informacji może wpłynąć na oryginalny charakter nagrania. W związku z tym stosuje się zarówno metody obiektywne, jak i subiektywne. Przykładem są testy preferencyjne, w których słuchacze oceniają jakość dźwięku pod kątem jego spójności i naturalności @16.

=== Etyczne aspekty dodawania nowych informacji do historycznych nagrań

Dodawanie nowych informacji do historycznych nagrań rodzi szereg pytań etycznych. Główna kwestia dotyczy tego, na ile możemy modyfikować oryginalny materiał, by nie zatracić jego autentyczności. Rozszerzanie pasma częstotliwościowego za pomocą AI i GAN musi być prowadzone z poszanowaniem dla oryginalnego dzieła, aby zachować jego integralność i nie wprowadzać zmian, które mogłyby zostać odebrane jako manipulacje oryginałem @17 @3.


== 4.3. Uzupełnianie brakujących fragmentów

=== Przyczyny i charakterystyka ubytków
Braki w nagraniach muzycznych mogą mieć różnorodne przyczyny, takie jak uszkodzenia fizyczne nośników, błędy w digitalizacji, czy celowe wycięcia fragmentów w procesie edycji. Charakterystyka tych ubytków jest równie zróżnicowana – od krótkich, niemal niezauważalnych przerw, po dłuższe fragmenty, które znacząco wpływają na integralność utworu muzycznego. W związku z tym, rekonstrukcja brakujących fragmentów stała się kluczowym zadaniem w konserwacji i restauracji nagrań dźwiękowych.

=== Metody AI do interpolacji brakujących fragmentów
W ostatnich latach znaczący postęp dokonał się w dziedzinie sztucznej inteligencji, szczególnie w kontekście interpolacji brakujących danych audio. Metody te wykorzystują zaawansowane modele uczenia maszynowego, które są zdolne do odtwarzania brakujących próbek dźwiękowych w sposób, który jest trudny do odróżnienia od oryginału. Na przykład, techniki oparte na modelach autoregresyjnych, takich jak Rekurencyjne Sieci Neuronowe (RNN) i Long Short-Term Memory (LSTM), umożliwiają przewidywanie brakujących próbek na podstawie istniejącego kontekstu dźwiękowego, co prowadzi do bardziej naturalnej rekonstrukcji @18.

=== Wykorzystanie kontekstu muzycznego w rekonstrukcji ubytków
Modele te mogą efektywnie wykorzystywać kontekst muzyczny, analizując struktury melodyczne, rytmiczne i harmoniczne, co pozwala na precyzyjne wypełnienie braków w sposób, który zachowuje spójność i naturalność nagrania. Ważnym aspektem jest tutaj także ocena spójności muzycznej rekonstruowanych fragmentów, która może być przeprowadzona zarówno subiektywnie, poprzez testy odsłuchowe, jak i obiektywnie, z wykorzystaniem narzędzi analitycznych @1.


== 4.4. Poprawa jakości mocno skompresowanych plików audio
Kompresja stratna, taka jak MP3, AAC, czy OGG, jest powszechnie stosowana w celu redukcji rozmiaru plików audio. Jednak proces ten nieodłącznie wiąże się z utratą pewnych informacji, co wpływa na jakość dźwięku. W szczególności mogą pojawić się artefakty, takie jak brakujące detale w wyższych częstotliwościach czy zniekształcenia perkusji, które negatywnie wpływają na odbiór muzyczny @8.

Aby przeciwdziałać tym problemom, rozwijane są techniki oparte na sztucznej inteligencji (AI). Jednym z obiecujących podejść jest zastosowanie głębokich sieci neuronowych, które mogą identyfikować i redukować artefakty kompresji. Przykładowo, modele oparte na architekturze U-Net czy Wave-U-Net są w stanie skutecznie poprawiać jakość dźwięku, szczególnie w przypadku nagrań mocno skompresowanych @6.

Zastosowanie Generatywnych Sieci Przeciwstawnych (GAN) otwiera nowe możliwości w odtwarzaniu detali utraconych podczas kompresji. GAN-y potrafią generować brakujące fragmenty sygnału audio w sposób realistyczny, co pozwala na znaczną poprawę jakości muzyki. Badania wykazują, że sieci GAN są szczególnie skuteczne w zwiększaniu rozdzielczości częstotliwościowej nagrań oraz w poprawie jakości dźwięku w skompresowanych plikach MP3 @11.

Istotną częścią tych procesów jest odpowiednie szkolenie modeli AI. Trening odbywa się na parach nagrań przed i po kompresji, co umożliwia modelom nauczenie się odtwarzania utraconych detali. Wyzwanie stanowi jednak generalizacja tych modeli na różne formaty kompresji, gdyż algorytmy mogą wykazywać różną skuteczność w zależności od typu kompresji. Dalsze badania są konieczne, aby zapewnić efektywne działanie tych technologii w szerokim spektrum formatów @10 @12.

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

