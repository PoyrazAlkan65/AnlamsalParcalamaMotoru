"""Ders analizi istemi (lesson-v1).

Köşe kuralları Admin courseGrowth.js / LMS scripts/growthAnalyzeCourses.js (program-v1) ile aynıdır;
fark: girdi dersin transkriptidir ve parça bazlı skor da istenir. Birini değiştirirseniz diğerlerini de gözden geçirin.
"""
from __future__ import annotations

import re

from . import settings

_TAGS = re.compile(r"<[^>]*>")


def strip_html(value) -> str:
    text = _TAGS.sub(" ", str(value or "")).replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def _clock(sec) -> str:
    if sec is None:
        return "--:--"
    s = int(float(sec))
    return f"{s // 60:02d}:{s % 60:02d}"


def build(lesson: dict, groups: list[dict], chunks: list[dict]) -> str:
    group_lines = "\n".join(f"- {g['Code']} ({g['GroupName']}): {g['DescriptionGroup']}" for g in groups)
    chunk_lines = "\n\n".join(
        f"[{c['Seq']}] ({_clock(c.get('StartSec'))}–{_clock(c.get('EndSec'))})\n{c['ChunkText'].strip()}" for c in chunks)
    desc = strip_html(lesson.get("LessonsDesc"))[:1000]

    return f"""Sen bir kişisel gelişim içeriği analistisin. Aşağıda bir eğitim programındaki tek bir dersin video transkripti, numaralı anlamsal parçalar halinde verilmiştir. Bu dersi izleyen bir katılımcının hangi gelişim köşelerini beslediğini belirle.

Görev:
1) Dersin tamamı için: etkiyi aşağıdaki 9 köşeye yüzde olarak dağıt.
   - Dersin katkı yaptığı bütün köşeleri seç; seçtiğin köşelerin yüzdeleri toplamı tam 100 olmalı.
   - Dersin ana odağına en yüksek yüzdeyi ver; ikincil ve dolaylı katkılara katkının büyüklüğü oranında düşük yüzde ver.
   - Transkriptte hiç karşılığı olmayan köşeye pay verme. Konuşmacının yalnızca örnek olarak geçtiği konular (ör. bir cümlelik değinme) köşe sayılmaz.
   - Her seçtiğin köşe için transkriptten somut dayanak gösteren, en fazla iki cümlelik Türkçe gerekçe yaz.
2) Her parça için: o parçanın açıkça beslediği en fazla {settings.MAX_CHUNK_GROUPS} köşeyi, 0 ile 1 arasında bir güçle (score) ver ve parçadan kısa bir dayanak (evidence, en fazla bir cümle, mümkünse kısa alıntı) yaz. Hiçbir köşeyi açıkça beslemeyen parça (giriş, selamlama, geçiş) için boş liste ver.
- Transkript otomatik ses tanımadan geldi; bozuk kelimeleri bağlamdan anlamlandır.
- Sadece verilen metne dayan. Sadece geçerli JSON döndür, başka metin yazma.

Köşeler (kod, ad ve ne ölçtüğü):
{group_lines}

Çıktı şeması:
{{"summary":"<dersin bir cümlelik özeti>","groups":[{{"code":"<KÖŞE_KODU>","percent":<1-100 tam sayı>,"reason":"<gerekçe>"}}],"chunks":[{{"seq":<parça no>,"groups":[{{"code":"<KÖŞE_KODU>","score":<0-1>,"evidence":"<dayanak>"}}]}}]}}

DERS
Program: {strip_html(lesson.get('CourseName'))}
Bölüm: {strip_html(lesson.get('SeasonName'))}
Ders: {strip_html(lesson.get('LessonsName'))}
{"Ders açıklaması: " + desc + chr(10) if desc else ""}
TRANSKRİPT PARÇALARI
{chunk_lines}"""
