# Learning-based Autofocus for Holography (Master Thesis)

## هدف المشروع
تدريب شبكة عصبية (CNN) لتقدير المسافة (z01) مباشرة من هولوغرام الأشعة السينية، كبديل لطرق التحسين البطيئة (Nelder-Mead).

## كيفية التشغيل
1. أنشئ البيئة: `conda env create -f environment.yml`
2. فعّل البيئة: `conda activate holophd`
3. شغّل التدريب: `python src/train.py --config configs/base.yaml`
