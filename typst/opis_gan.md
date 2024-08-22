W ramach badania nad zastosowaniem Generatywnych Sieci Przeciwstawnych (GAN) w rekonstrukcji nagrań dźwiękowych, opracowano zaawansowaną architekturę sieci neuronowej. Model składa się z generatora i dyskryminatora, które współpracują w procesie uczenia się rekonstrukcji uszkodzonych nagrań muzyki klasycznej.
Generator wykorzystuje architekturę typu encoder-decoder z blokami rezydualnymi w części centralnej. Encoder składa się z czterech warstw konwolucyjnych, które stopniowo zmniejszają wymiarowość danych wejściowych, jednocześnie zwiększając liczbę kanałów. Bottleneck zawiera trzy bloki rezydualne, które pomagają w zachowaniu informacji przestrzennych. Decoder, składający się z czterech warstw dekonwolucyjnych, rekonstruuje oryginalny kształt danych, korzystając z połączeń skip między odpowiadającymi sobie warstwami encodera i decodera. Taka struktura pozwala na efektywne przetwarzanie informacji na różnych poziomach abstrakcji.
Dyskryminator zbudowany jest z pięciu bloków konwolucyjnych, które stopniowo redukują wymiarowość danych wejściowych, zwiększając jednocześnie liczbę kanałów. Końcowa warstwa konwolucyjna generuje pojedynczą wartość, reprezentującą ocenę autentyczności wejścia.
W celu poprawy stabilności treningu i jakości generowanych wyników, zastosowano szereg zaawansowanych technik. Wykorzystano normalizację spektralną w warstwach konwolucyjnych, co pomaga w kontrolowaniu dynamiki gradientów. Wprowadzono również szum instancji (instance noise) z mechanizmem stopniowego zmniejszania jego intensywności, co wspomaga proces uczenia się w początkowych fazach treningu.
Model wykorzystuje złożoną funkcję straty, składającą się z wielu komponentów. Obejmują one stratę adwersarialną, stratę zawartości, stratę spektralnej zbieżności, stratę płaskości widmowej, stratę uwzględniającą fazę, stratę multi-rozdzielczą STFT, stratę percepcyjną, stratę czasowo-częstotliwościową oraz stratę SNR. Każdy z tych komponentów ma przypisaną wagę, która może być dostosowywana w celu zoptymalizowania procesu uczenia.
W trakcie treningu zastosowano technikę akumulacji gradientów, co pozwala na efektywne zwiększenie rozmiaru batcha bez zwiększania zużycia pamięci. Wprowadzono również adaptacyjne dostosowywanie częstotliwości aktualizacji dyskryminatora oraz dynamiczne dostosowywanie współczynnika uczenia w zależności od wartości funkcji straty.
Wyniki badania wykazały, że model jest w stanie częściowo usunąć trzaski charakterystyczne dla płyt winylowych, co widoczne jest na spektrogramach STFT. Jednakże, wygenerowane utwory zawierały nadmierne szumy i zniekształcenia, co uniemożliwiło ich subiektywną ocenę poprzez odsłuch. Mimo to, badanie to stanowi istotny krok w kierunku opracowania efektywnych metod rekonstrukcji historycznych nagrań dźwiękowych i otwiera drogę do dalszych badań nad generatywnym odszumianiem nagrań oraz optymalizacją funkcji straty dla celów rekonstrukcyjnych i poprawy jakości dźwięku.


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