"""SedaUlgen LMS ders analizi mikroservisi.

Ders videosu → transkript (Whisper ya da içe aktarma) → anlamsal parçalar → yapay zekâ ile
9 gelişim köşesine parça ve ders bazlı etki → LMS veritabanı (sprint22 prosedürleri).
Proje kökünden çalıştırılır:
  python -m services.lesson_analysis.worker      # kuyruk işleyici
  uvicorn services.lesson_analysis.api:app       # HTTP API
  python -m services.lesson_analysis.import_transcripts --course 33 --dir ...
"""
