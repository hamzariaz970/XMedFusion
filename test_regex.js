const text = `[FINAL REPORT]:
FINDINGS:
Present findings include cardiomegaly. No pleural effusion is identified. No pneumothorax is identified. No focal infiltrate is identified. No focal consolidation is identified. No atelectasis is identified. No acute osseous abnormality is identified.

IMPRESSION:
Cardiomegaly.

RECOMMENDATIONS:
Radiologist review and clinical correlation are recommended because abnormal findings are supported by the image evidence.

LABELS:
Cardiomegaly`;

const findingsMatch = text.match(/FINDINGS:([\s\S]*?)(?=IMPRESSIONS?:|$)/i);
const impressionMatch = text.match(/IMPRESSIONS?:([\s\S]*?)(?=RECOMMENDATIONS?:|LABELS:|$)/i);
const recommendationMatch = text.match(/RECOMMENDATIONS?:([\s\S]*?)(?=LABELS:|$)/i);
const labelsMatch = text.match(/LABELS:([\s\S]*?)$/i);

console.log('Findings:', findingsMatch ? findingsMatch[1].trim() : 'NOT FOUND');
console.log('Impression:', impressionMatch ? impressionMatch[1].trim() : 'NOT FOUND');
console.log('Recommendation:', recommendationMatch ? recommendationMatch[1].trim() : 'NOT FOUND');
console.log('Labels:', labelsMatch ? labelsMatch[1].trim() : 'NOT FOUND');
