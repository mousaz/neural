function loadDataFromFile(file) {
  return new Promise((resolve) => {
    const fileReader = new FileReader();
    fileReader.addEventListener("load", () => {
      const lines = fileReader.result.split("\n");
      const data = { rowsCount: lines.length };
      for (const line of lines) {
        const values = line.split(",");
        data.columnsCount = values.length;
        for (let i = 0; i < values.length; i++) {
          if (!data[i]) data[i] = [];
          data[i].push(values[i].trim());
        }
      }

      resolve(data);
    });

    fileReader.readAsText(file);
  });
}
