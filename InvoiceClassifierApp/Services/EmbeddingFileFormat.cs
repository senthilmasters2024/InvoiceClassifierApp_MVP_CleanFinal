using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InvoiceClassifierApp.Services
{
    public class EmbeddingFileFormat
    {
        public string Filename { get; set; }

        public string? Label { get; set; }
        public float[] Vector { get; set; }
    }

}
