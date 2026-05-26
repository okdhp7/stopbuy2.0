import type { AlternativeProduct } from "@/hooks/useStopBuyWS";

export const PRODUCT_PLACEHOLDER_IMAGE =
  "https://d2xsxph8kpxj0f.cloudfront.net/310519663632451254/8SbY3RBNCHrKBA7jPh2CJe/stopbuy_product_placeholder-ZSAsS7GCJQb9MZnQ4RoFR7.webp";

type ProductImageRule = {
  keywords: string[];
  imageUrl: string;
};

const EXACT_PRODUCT_IMAGE_RULES: ProductImageRule[] = [
  {
    keywords: ["galaxy s24 fe", "s24 fe"],
    imageUrl: "https://fdn2.gsmarena.com/vv/bigpic/samsung-galaxy-s24-fe.jpg",
  },
  {
    keywords: ["iphone 15"],
    imageUrl: "https://fdn2.gsmarena.com/vv/pics/apple/apple-iphone-15-1.jpg",
  },
  {
    keywords: ["pixel 8a"],
    imageUrl: "https://fdn2.gsmarena.com/vv/bigpic/google-pixel-8a.jpg",
  },
  {
    keywords: ["xiaomi 14t"],
    imageUrl: "https://fdn2.gsmarena.com/vv/bigpic/xiaomi-14t.jpg",
  },
  {
    keywords: ["oneplus 12r"],
    imageUrl: "https://fdn2.gsmarena.com/vv/bigpic/oneplus-12r.jpg",
  },
  {
    keywords: ["nothing phone 2a"],
    imageUrl: "https://fdn2.gsmarena.com/vv/bigpic/nothing-phone-2a.jpg",
  },
  {
    keywords: ["macbook air m3"],
    imageUrl:
      "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/mba13-m3-midnight-gallery1-202402?wid=900&hei=600&fmt=jpeg&qlt=90&.v=1707416815196",
  },
  {
    keywords: ["sony wh-1000xm5", "wh-1000xm5"],
    imageUrl: "https://mma.prnewswire.com/media/1816079/Sony_WH_1000XM5_headphones.jpg?p=publish",
  },
  {
    keywords: ["airpods pro 2"],
    imageUrl:
      "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/MQD83?wid=900&hei=600&fmt=png-alpha&.v=1660803972361",
  },
  {
    keywords: ["bose quietcomfort ultra"],
    imageUrl:
      "https://assets.bosecreative.com/transform/775c3e9a-fcd1-489f-a2f7-a57ac66464e1/SF_QCUH_deepplum_gallery_1_816x612_x2?io=width%3A816%2Cheight%3A667%2Ctransform%3Afit&quality=90",
  },
  {
    keywords: ["galaxy buds3 pro", "buds3 pro"],
    imageUrl:
      "https://api.samsungmobilepress.com/api/v1/file/A883670379D5943613666FC47FDF336B969C8DAB0EE5DB9678DE17F1835EB12061AC10F8380C857FB887A4DA4D5CFD522B5EE84E4BE72A7B91D06F877E9ADFD4AF4A4E99108B76E53EE5FD7DB9A33BF4EF185AD10D51B1C0FEDFF15A6BAE9B98F7E8754FA83F81FA804EAC2ED054D34AA29D19123081709FE794338202CFD7F1",
  },
  {
    keywords: ["jbl tour pro 3", "tour pro 3"],
    imageUrl:
      "https://jblstore.co.id/wp-content/uploads/2024/11/01.LS-JBL-Tour-Pro-3-Product-Image-Case-Open-Black-600x600.webp",
  },
];

function normalize(value?: string | number | null): string {
  return String(value ?? "").toLowerCase().replace(/\s+/g, " ").trim();
}

export function getAlternativeProductImage(product: AlternativeProduct): string {
  const searchableText = normalize(
    [product.name, product.brand, product.category].filter(Boolean).join(" "),
  );

  const imageUrl = product.image_url?.trim();
  if (imageUrl && !imageUrl.includes("source.unsplash.com")) {
    return imageUrl;
  }

  const exactMatch = EXACT_PRODUCT_IMAGE_RULES.find((rule) =>
    rule.keywords.some((keyword) => searchableText.includes(keyword)),
  );

  if (exactMatch) {
    return exactMatch.imageUrl;
  }

  return PRODUCT_PLACEHOLDER_IMAGE;
}
