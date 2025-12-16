using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FixClientLite
{
    public class Trace
    {
        public int RawFeedId { get; set; }
        public DateTime CreatedDate { get; set; }
        /// <summary>Gets or sets the FINRA created date.</summary>
        public DateTime FinraCreatedDate { get; set; }

        /// <summary>Gets or sets the MBS message type.</summary>
        public TraceMessageType MbsMessageType { get; set; }

        /// <summary>Gets or sets the FINRA trade type id.</summary>
        public TraceOrigin FinraTradeTypeId { get; set; }

        /// <summary>Gets or sets the exchange.</summary>
        
        public string Exchange { get; set; }

        /// <summary>Gets or sets the RDID.</summary>
       
        public string RDID { get; set; }

        /// <summary>Gets or sets the RDID.</summary>
       
        public string RDIDDescription { get; set; }

        /// <summary>Gets or sets the cusip.</summary>
       
        public string Cusip { get; set; }

       
        public string Description { get; set; }

        /// <summary>Gets or sets the symbol.</summary>
        
        public string Symbol { get; set; }

        /// <summary>Gets or sets the quantity.</summary>
        public long? Quantity { get; set; }

        /// <summary>Gets or sets the price.</summary>
        public double? Price { get; set; }

        /// <summary>Gets or sets the execution date time.</summary>
        public DateTime? ExecutionDateTime { get; set; }

        /// <summary>Gets or sets the settlement date.</summary>
        public DateTime? SettlementDate { get; set; }

        /// <summary>Gets or sets the quantity indicator.</summary>
       
        public string QuantityIndicator { get; set; }

        /// <summary>Gets or sets the remuneration.</summary>
       
        public string Remuneration { get; set; } // Renamed from CommissionIndicator

        /// <summary>Gets or sets the special price indicator.</summary>
       
        public string SpecialPriceIndicator { get; set; }

        /// <summary>Gets or sets the reporting party side.</summary>
        public TraceReportingPartySide? ReportingPartySide { get; set; } // Renamed from ReportingPartySide

        /// <summary>Gets or sets the sale condition 3.</summary>
       
        public string SaleCondition3 { get; set; }

        /// <summary>Gets or sets the sale condition 4.</summary>
       
        public string SaleCondition4 { get; set; }

        /// <summary>Gets or sets the change indicator.</summary>
       
        public string ChangeIndicator { get; set; }

        /// <summary>Gets or sets the as of indicator.</summary>
       
        public string AsOfIndicator { get; set; }

        /// <summary>Gets or sets the factor.</summary>
        public double? Factor { get; set; }

        /// <summary>Gets or sets the BSYM.</summary>        
        public string Bsym { get; set; }

        /// <summary>Gets or sets the sub product type.</summary>
        
        public string SubProductType { get; set; }       

        /// <summary>Gets or sets the reporting party type.</summary>
       
        public string ReportingPartyType { get; set; } // new since v. 2.2 -> MBSS 1815

        /// <summary>Gets or sets the contra party type.</summary>
       
        public string ContraPartyType { get; set; } // new since v. 2.2 -> MBSS 1815

        /// <summary>Gets or sets the original message sequence number.</summary>
        public long? OriginalMessageSeqNumber { get; set; } //new since v.  2.2-> MBS 2141

        /// <summary>Gets or sets the message sequence number.</summary>
        public long? MessageSeqNumber { get; set; } //new since v.  2.2-> MBS 2141

        /// <summary>Gets or sets the original dissemination date.</summary>
        public DateTime? OriginalDisseminationDate { get; set; }

        /// <summary>Gets or sets the ATS indicator.</summary>
       
        public string AtsIndicator { get; set; } // new since v. 2.4 -> MBSS 4639

        /// <summary>Gets or sets the dissemination date.</summary>
        public DateTime? DisseminationDate { get; set; }

        /// <summary>Gets or sets the high price.</summary>
        public double? HighPrice { get; set; }

        /// <summary>Gets or sets the low price.</summary>
        public double? LowPrice { get; set; }

        /// <summary>Gets or sets the last sale price.</summary>
        public double? LastSalePrice { get; set; }

        /// <summary>
        /// Entire FIX Message
        /// </summary>
        public string RawFixMessageAsXml { get; set; }

              

        /// <summary>
        /// Data received by client (BD datetime)
        /// </summary>
        public string DateInserted { get; set; }

        private static string Delimiter = ",";
        
        /// <summary>
        /// Header for CSV Appender
        /// </summary>
        /// <returns></returns>
        public static string GetHeader()
        {
            return "RawFeedId,CreatedDate,FinraCreatedDate,MbsMessageType,FinraTradeTypeId,Exchange,RDID,Cusip,Quantity,Price,ExecutionDateTime,SettlementDate,Factor,Bsym,SubProductType,OriginalMessageSeqNumber,MessageSeqNumber,OriginalDisseminationDate,DisseminationDate,HighPrice,LowPrice,LastSalePrice,QuantityIndicator,Remuneration,SpecialPriceIndicator,ReportingPartySide,SaleCondition3,SaleCondition4,ChangeIndicator,AsOfIndicator,ReportingPartyType,ContraPartyType,AtsIndicator,Symbol,RawFixMessageAsXml,DateInserted";
        }

        /// <summary>
        /// Line fo CSV appender
        /// </summary>
        /// <returns></returns>
        public string GetCsvLine()
        {
            StringBuilder sb = new StringBuilder(); 
            sb.Append(this.RawFeedId);
            sb.Append(Delimiter);
            sb.Append(this.CreatedDate);
            sb.Append(Delimiter);
            sb.Append(this.FinraCreatedDate);
            sb.Append(Delimiter);
            sb.Append(this.MbsMessageType);
            sb.Append(Delimiter);
            sb.Append(this.FinraTradeTypeId);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.Exchange) ? this.Exchange.Contains(Delimiter) ? string.Concat("\"", this.Exchange, "\"") : this.Exchange : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.RDID) ? this.RDID.Contains(Delimiter) ? string.Concat("\"", this.RDID, "\"") : this.RDID : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.Cusip) ? this.Cusip.Contains(Delimiter) ? string.Concat("\"", this.Cusip, "\"") : this.Cusip : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.Quantity.HasValue ?  this.Quantity.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.Price.HasValue ? this.Price.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.ExecutionDateTime.HasValue ? this.ExecutionDateTime.Value.ToString() : string.Empty);
            sb.Append(Delimiter); 
            sb.Append(this.SettlementDate.HasValue ? this.SettlementDate.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.Factor.HasValue ? this.Factor.Value.ToString() : string.Empty);
            sb.Append(Delimiter); 
            sb.Append(!string.IsNullOrEmpty(this.Bsym) ? this.Bsym.Contains(Delimiter) ? string.Concat("\"", this.Bsym, "\"") : this.Bsym : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.SubProductType) ? this.SubProductType.Contains(Delimiter) ? string.Concat("\"", this.SubProductType, "\"") : this.SubProductType : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.OriginalMessageSeqNumber.HasValue ? this.OriginalMessageSeqNumber.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.MessageSeqNumber.HasValue ? this.MessageSeqNumber.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.OriginalDisseminationDate.HasValue ? this.OriginalDisseminationDate.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.DisseminationDate.HasValue ? this.DisseminationDate.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.HighPrice.HasValue ? this.HighPrice.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.LowPrice.HasValue ? this.LowPrice.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.LastSalePrice.HasValue ? this.LastSalePrice.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.QuantityIndicator) ? this.QuantityIndicator.Contains(Delimiter) ? string.Concat("\"", this.QuantityIndicator, "\"") : this.QuantityIndicator : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.Remuneration) ? this.Remuneration.Contains(Delimiter) ? string.Concat("\"", this.Remuneration, "\"") : this.Remuneration : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.SpecialPriceIndicator) ? this.SpecialPriceIndicator.Contains(Delimiter) ? string.Concat("\"", this.SpecialPriceIndicator, "\"") : this.SpecialPriceIndicator : string.Empty);
            sb.Append(Delimiter);
            sb.Append(this.ReportingPartySide.HasValue ? this.ReportingPartySide.Value.ToString() : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.SaleCondition3) ? this.SaleCondition3.Contains(Delimiter) ? string.Concat("\"", this.SaleCondition3, "\"") : this.SaleCondition3 : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.SaleCondition4) ? this.SaleCondition4.Contains(Delimiter) ? string.Concat("\"", this.SaleCondition4, "\"") : this.SaleCondition4 : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.ChangeIndicator) ? this.ChangeIndicator.Contains(Delimiter) ? string.Concat("\"", this.ChangeIndicator, "\"") : this.ChangeIndicator : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.AsOfIndicator) ? this.AsOfIndicator.Contains(Delimiter) ? string.Concat("\"", this.AsOfIndicator, "\"") : this.AsOfIndicator : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.ReportingPartyType) ? this.ReportingPartyType.Contains(Delimiter) ? string.Concat("\"", this.ReportingPartyType, "\"") : this.ReportingPartyType : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.ContraPartyType) ? this.ContraPartyType.Contains(Delimiter) ? string.Concat("\"", this.ContraPartyType, "\"") : this.ContraPartyType : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.AtsIndicator) ? this.AtsIndicator.Contains(Delimiter) ? string.Concat("\"", this.AtsIndicator, "\"") : this.AtsIndicator : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.Symbol) ? this.Symbol.Contains(Delimiter) ? string.Concat("\"", this.Symbol, "\"") : this.Symbol : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.RawFixMessageAsXml) ? this.RawFixMessageAsXml.Contains(Delimiter) ? string.Concat("\"", this.RawFixMessageAsXml, "\"") : this.RawFixMessageAsXml : string.Empty);
            sb.Append(Delimiter);
            sb.Append(!string.IsNullOrEmpty(this.DateInserted) ? this.DateInserted.Contains(Delimiter) ? string.Concat("\"", this.DateInserted, "\"") : this.DateInserted : string.Empty);
            return sb.ToString();
        }
    }

    [Serializable]
    public enum TraceMessageType
    {
        Trade = 0, // T in historical

        Correction = 1, // R in historical

        Cancel = 2, // X in historical

        CanceledCorrection = 3, // C in historical

        Reversal = 4 // Y in historical; a transaction that has been reverse more than 20 days after it was input
    }
    [Serializable]
    public enum TraceOrigin
    {
        Import = 0,

        Current = 1,

        Historical = 2
    }

    [Serializable]
    public enum TraceReportingPartySide : int
    {
        [Description("Dealer bought from customer")]
        DealerBoughtFromCustomer = 0,

        [Description("Dealer sold to customer")]
        DealerSoldToCustomer = 1,

        [Description("Dealer-to-dealer")]
        DealerToDealer = 2
    }
}
