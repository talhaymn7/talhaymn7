const sharp = require("sharp");
const path = require("path");

const SIZE = 600;

async function run() {
  const jobs = [
    ["img/archlinux.svg", "logo_archlinux.png"],
    ["img/Node.js_logo.svg", "logo_nodejs.png"],
  ];
  for (const [src, dst] of jobs) {
    await sharp(src, { density: 384 })
      .resize(SIZE, SIZE, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toFile(dst);
    console.log(`${src} -> ${dst}`);
  }
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
